"""
评估脚本: evaluate_adaptive_parallel.py
功能：加载训练好的 Adaptive RL (PPO) 模型和冻结的 Diffusion Policy，进行并行评估。
"""
import os
# 强制使用 EGL 渲染后端
os.environ['MUJOCO_GL'] = 'egl'
# 禁用一些可能导致冲突的渲染特性
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# 针对 NVIDIA 驱动的特殊设置，防止多进程冲突
os.environ['EGL_PBUFFER_WIDTH'] = '160'
os.environ['EGL_PBUFFER_HEIGHT'] = '160'
# 限制 ffmpeg 的线程数，防止它在编码视频时抢走所有 CPU 资源导致其他进程渲染超时
os.environ['IMAGEIO_FFMPEG_THREADS'] = '2'

import argparse
import sys
import os
import torch
import numpy as np
import json
import random
from datetime import datetime
from functools import partial
from collections import defaultdict

# --- 1. 路径与环境修复 (与训练脚本保持一致) ---
try:
    import robosuite
    if not hasattr(robosuite, "make"):
        import robosuite.environments.base as base
        import robosuite.environments.manipulation
        from robosuite.utils import binding_utils
    if not hasattr(robosuite, "__version__"):
        robosuite.__version__ = "1.4.1"
except ImportError:
    pass

# 修正 Python 路径以找到 wuhao 文件夹
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "../..")) 
sys.path.append(repo_root)

# 导入必要的库
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.env_utils as EnvUtils

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
    from stable_baselines3.common.buffers import RolloutBuffer
    from stable_baselines3.common.utils import obs_as_tensor
except ImportError:
    print("请安装 stable-baselines3: pip install stable-baselines3[extra]")
    sys.exit(1)

# 导入自定义 Wrapper
try:
    from wuhao.adaptive_env_utils_parallel import BatchedAdaptiveWrapper, RobomimicToGymWrapper, DSRLStylePolicy, InternalVariableChunkWrapper
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(current_dir, "../../")))
    from wuhao.adaptive_env_utils_parallel import BatchedAdaptiveWrapper, RobomimicToGymWrapper, DSRLStylePolicy, InternalVariableChunkWrapper


# =============================================================================
# 2. 必须重新定义 SMDP 类以支持模型加载 (Pickle 需要类定义)
# =============================================================================

class SMDPRolloutBuffer(RolloutBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.durations = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def reset(self) -> None:
        self.durations = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        super().reset()

    def add(self, obs, action, reward, episode_start, value, log_prob, durations=None) -> None:
        if durations is not None:
            self.durations[self.pos] = np.array(durations).copy()
        super().add(obs, action, reward, episode_start, value, log_prob)
    
    # 评估时不需要 compute_returns_and_advantage，但为了类完整性保留

class SMDPPPO(PPO):
    def __init__(self, *args, **kwargs):
        kwargs["rollout_buffer_class"] = SMDPRolloutBuffer
        super().__init__(*args, **kwargs)

# =============================================================================
# 3. 环境构建函数 (Eval 版)
# =============================================================================

def make_eval_env(env_meta, env_id, args, render_offscreen=False, save_data=False):
    # [新增] 让不同环境错开初始化时间，避免抢夺显卡驱动
    import time
    time.sleep(env_id * 2)

    # 设置随机种子
    base_seed = args.seed
    worker_seed = base_seed + env_id

    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    # 路径修复 (防止子进程找不到 robosuite)
    robosuite_dir = "/home/wuhao/jobspace/robosuite"
    if robosuite_dir not in sys.path:
        sys.path.insert(0, robosuite_dir)
    
    import robosuite
    import robosuite.environments.base as suite_base
    if not hasattr(robosuite, "make"):
        robosuite.make = suite_base.make
    if not hasattr(robosuite, "__version__"):
        robosuite.__version__ = "1.4.1"

    # 注册环境
    try:
        from robosuite.environments.manipulation.tool_hang import ToolHang
        suite_base.register_env(ToolHang)
    except Exception:
        pass

    # 初始化 ObsUtils
    import robomimic.utils.obs_utils as ObsUtils
    all_keys = env_meta.get("all_obs_keys", [])
    if not all_keys:
        all_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
    
    obs_specs = {"obs": {"low_dim": all_keys, "rgb": []}}
    try:
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_specs)
    except Exception:
        ObsUtils.initialize_obs_utils_with_obs_specs({"low_dim": all_keys})

    # 创建原始环境
    import robomimic.utils.env_utils as EnvUtils
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=render_offscreen, 
        use_image_obs=False
    )

    # 包装成 Gym
    env = RobomimicToGymWrapper(env)

    # 包装变长执行器 (核心)
    # 评估时，视频保存逻辑由 InternalVariableChunkWrapper 处理
    # 它会自动将成功的存为 _success.mp4，失败的存为 _failed.mp4
    env = InternalVariableChunkWrapper(
        env, 
        reward_offset=1.0, 
        max_steps=args.horizon, # 使用传入的 horizon
        save_videos=render_offscreen, # 如果 render_offscreen 为 True，则开启录制
        save_data=save_data,          # 控制数据 [新增]
        video_dir=args.video_dir,
        env_id=env_id,
        gamma=0.999 # 评估时 gamma 不影响策略执行，但 Wrapper 需要参数
    )
    
    return env

# =============================================================================
# 4. 主评估逻辑
# =============================================================================

def evaluate(args):
    # [新增] 强制单进程逻辑
    if args.save_data and args.n_envs > 1:
        print(f"\n[WARNING] 检测到开启了 --save_data (保存带图片的PKL)。")
        print(f"[WARNING] 为了防止并发渲染导致的显存缓冲区冲突（雪花图、翻转图），强制将 n_envs 设为 1。")
        args.n_envs = 1

    # [新增] 强制 PyTorch 使用确定性算法
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # 这一行非常关键，但也可能导致运行变慢
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 如果 PyTorch 版本支持：
    torch.use_deterministic_algorithms(True)


    # 设置设备
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    print(f"[INFO] Using Device: {device}")

    # 1. 加载 Base Agent (Diffusion Policy)
    print(f"[INFO] Loading Base Agent: {args.agent}")
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=args.agent, 
        device=device, 
        verbose=False
    )
    
    # 冻结 Base Agent
    algo_instance = policy.policy 
    pred_h = algo_instance.algo_config.horizon.prediction_horizon
    with algo_instance.algo_config.values_unlocked():
        algo_instance.algo_config.horizon.action_horizon = pred_h
    if hasattr(algo_instance, 'ac_horizon'):
        algo_instance.ac_horizon = pred_h
    
    algo_instance.nets.eval()
    for param in algo_instance.nets.parameters():
        param.requires_grad = False
    algo_instance.use_action_scheduler = False 
    algo_instance.use_ood_monitor = False

    # 2. 准备环境配置
    env_meta = ckpt_dict["env_metadata"]
    # 覆盖 Horizon
    if args.horizon:
        env_meta["env_kwargs"]["horizon"] = args.horizon
    
    print(f"[INFO] Evaluation Horizon: {env_meta['env_kwargs']['horizon']}")

    # [修改点] 确定是否需要开启离屏渲染
    # 只要需要保存视频 OR 需要在 pkl 中存图，就必须设为 True
    need_render = args.save_video or args.save_data # [新增逻辑]

    # 3. 创建并行环境
    # 如果 save_video 为 True，我们需要 render_offscreen=True
    env_fns = [partial(make_eval_env, env_meta, i, args, 
                       render_offscreen=need_render, # 是否存 MP4
                       save_data=args.save_data)         # 是否存 PKL [新增]
               for i in range(args.n_envs)]
    
    print(f"[INFO] Launching {args.n_envs} parallel environments...")
    # 使用 spawn 启动
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    raw_venv = SubprocVecEnv(env_fns)

    # 4. 包装 Adaptive Wrapper (主进程)
    venv = BatchedAdaptiveWrapper(
        venv=raw_venv,
        policy=policy,
        max_chunk_len=args.max_chunk_len,
        device=device,
        reward_offset=0.0,
        resample_penalty=0.0,
        action_history_len=args.action_history_len
    )
    
    # 使用 VecMonitor 方便统计
    venv = VecMonitor(venv)
    # 如果训练时使用了 VecNormalize，加载模型时会自动处理，或者需要手动加载 stats
    # 这里我们创建一个新的 VecNormalize，但不进行训练更新
    venv = VecNormalize(venv, norm_obs=False, norm_reward=False, training=False, clip_obs=10.)


    # 5. [修改] 根据参数决定加载 RL 模型还是使用固定策略
    model = None

    if args.fixed_chunk_len is not None:
        print(f"[INFO] Mode: FIXED Chunk Length = {args.fixed_chunk_len}")
        # 简单检查一下
        assert args.fixed_chunk_len <= args.max_chunk_len, "Fixed chunk length cannot exceed max_chunk_len"
    elif args.rl_model is not None:
        print(f"[INFO] Mode: RL Policy from {args.rl_model}")
        model = SMDPPPO.load(args.rl_model, env=venv, device=device)
    else:
        raise ValueError("Please provide either --rl_model or --fixed_chunk_len")

    # 6. 开始评估循环
    print(f"[INFO] Starting Evaluation for {args.n_rollouts} episodes...")
    
    total_episodes = 0
    success_count = 0
    returns = []
    lengths = []
    action_counts = defaultdict(int)

    obs = venv.reset()
    
    # 由于是并行环境，我们需要追踪每个环境完成的次数
    env_dones = [0] * args.n_envs
    
    # 循环直到收集足够的 episodes
    while total_episodes < args.n_rollouts:
        # [修改] 动作生成逻辑
        if args.fixed_chunk_len is not None:
            # 如果是固定模式，生成全为 fixed_chunk_len 的数组
            # venv.num_envs 获取当前并行环境数量
            action = np.full(venv.num_envs, args.fixed_chunk_len, dtype=int)
        else:
            # 如果是 RL 模式，使用模型预测
            action, _ = model.predict(obs, deterministic=True)
            # action, _ = model.predict(obs, deterministic=False)

            # [新增] 提取分布并传给 Wrapper
            # 仅当需要保存数据时才做这个计算（省资源）
            if args.save_data:
                obs_tensor = torch.as_tensor(obs).to(device)
                with torch.no_grad():
                    # SB3 PPO 策略提取分布概率
                    dist = model.policy.get_distribution(obs_tensor)
                    probs = dist.distribution.probs.cpu().numpy()
                venv.set_next_step_dists(probs)
            else:
                venv.set_next_step_dists(None)
        
        # 统计动作分布
        for a in action:
            action_counts[int(a)] += 1
            
        obs, rewards, dones, infos = venv.step(action)
        
        for i, done in enumerate(dones):
            if done:
                # 只有当还需要收集数据时才记录
                if total_episodes < args.n_rollouts:
                    total_episodes += 1
                    info = infos[i]
                    
                    # 获取真实返回值 (因为 VecNormalize 可能会归一化 reward)
                    ep_ret = info.get('episode', {}).get('r', 0)
                    ep_len = info.get('episode', {}).get('l', 0)
                    
                    # 检查成功 (BatchedAdaptiveWrapper/InternalVariableChunkWrapper 应该传递 is_success)
                    # Robomimic 环境通常在 info['is_success']
                    # 注意：VecMonitor 可能会把 info 包装一下，原始 info 在 terminal_info 里
                    
                    # 尝试从 info 直接获取，或者从 terminal_observation 附近的 info 获取
                    is_success = False
                    if "is_success" in info:
                         is_success = info["is_success"]
                    # 有时 SB3 会把原始 info 放在 'terminal_info' 或直接合并
                    # InternalVariableChunkWrapper 会把 is_success 放入 info
                    
                    if is_success:
                        success_count += 1
                    
                    returns.append(ep_ret)
                    lengths.append(ep_len)
                    
                    print(f"Episode {total_episodes}/{args.n_rollouts}: Success={is_success}, Return={ep_ret:.2f}, Length={ep_len}")

    # 7. 汇总结果
    success_rate = success_count / total_episodes
    avg_return = np.mean(returns)
    avg_length = np.mean(lengths)
    
    print("\n" + "="*40)
    print(f"EVALUATION RESULTS ({args.n_rollouts} episodes)")
    print("="*40)
    print(f"Success Rate: {success_rate * 100:.2f}%")
    print(f"Avg Return:   {avg_return:.4f}")
    print(f"Avg Length:   {avg_length:.2f}")
    print("-" * 40)
    print("Action Distribution (Chunk Lengths):")
    total_actions = sum(action_counts.values())
    sorted_actions = sorted(action_counts.keys())
    for k in sorted_actions:
        count = action_counts[k]
        ratio = count / total_actions
        print(f"  Length {k:2d}: {count:5d} ({ratio*100:.1f}%)")
    print("="*40)

    # 保存结果到 JSON
    results = {
        "success_rate": success_rate,
        "avg_return": float(avg_return),
        "avg_length": float(avg_length),
        "action_distribution": {k: int(v) for k, v in action_counts.items()},
        "args": vars(args)
    }
    
    json_path = os.path.join(args.video_dir, "eval_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[INFO] Results saved to {json_path}")

    venv.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 必需参数
    parser.add_argument("--agent", type=str, required=True, help="Base Diffusion Policy checkpoint (.pth)")

    # [修改] rl_model 改为可选 (default=None)，去掉 required=True
    parser.add_argument("--rl_model", type=str, default=None, help="Trained RL PPO checkpoint (.zip)")
    # [新增] 用于指定固定 Chunk 长度
    parser.add_argument("--fixed_chunk_len", type=int, default=None, help="If set, ignore RL policy and use this fixed chunk length")
    
    # 评估配置
    parser.add_argument("--n_rollouts", type=int, default=50, help="Total evaluation episodes")
    parser.add_argument("--n_envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--horizon", type=int, default=600, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=100, help="Random seed")
    
    # 必须与训练时一致的参数
    parser.add_argument("--max_chunk_len", type=int, default=32, help="Must match training config")
    parser.add_argument("--action_history_len", type=int, default=0, help="Must match training config")
    
    # 视频保存
    parser.add_argument("--save_video", action="store_true", help="Enable video recording")
    parser.add_argument("--video_dir", type=str, default="eval_videos", help="Directory to save videos and results")

    parser.add_argument("--save_data", action="store_true", help="Enable data logging (PKL: state, action, dist)") # [新增]

    args = parser.parse_args()
    
    # 自动创建输出目录
    os.makedirs(args.video_dir, exist_ok=True)
    
    evaluate(args)