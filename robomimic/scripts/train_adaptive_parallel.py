"""
整合版 train_adaptive.py
功能：加载冻结的预训练 Diffusion Policy，通过 PPO 训练 RL Agent 动态选取 Action Chunk 长度。
"""





import argparse
import sys
import os

# --- 【强制置顶补丁】 ---
try:
    import robosuite
    # 如果发现是空壳，强制加载核心组件
    if not hasattr(robosuite, "make"):
        import robosuite.environments.base as base
        import robosuite.environments.manipulation
        from robosuite.utils import binding_utils
    if not hasattr(robosuite, "__version__"):
        robosuite.__version__ = "1.4.1"
except ImportError:
    pass
# ---------------------

import torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import numpy as np
from collections import deque
from datetime import datetime

# from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor # 导入 VecMonitor
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import VecVideoRecorder # 导入视频记录器
from stable_baselines3.common.logger import TensorBoardOutputFormat
import torch.nn as nn


# --- 新增：强制修复 robosuite 版本属性缺失问题 ---
# try:
#     import robosuite
#     if not hasattr(robosuite, "__version__"):
#         # 赋予一个默认的 v1 版本号，让 robomimic 能够继续执行
#         robosuite.__version__ = "1.4.1" 
#         print("[INFO] 手动为 robosuite 模块添加了 __version__ 属性")
# except ImportError:
#     pass
# -----------------------------------------------

# 兼容性导入：处理新旧版 Gym 冲突
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

# =============================================================================
# 1. 路径修正：确保 Python 能找到 robomimic 库和根目录下的 wuhao 文件夹
# =============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
# 脚本位于 robomimic/robomimic/scripts/，向上三级到达仓库根目录以找到 wuhao
repo_root = os.path.abspath(os.path.join(current_dir, "../../..")) 
sys.path.append(repo_root)

# =============================================================================
# 2. 依赖导入 (SB3 & Robomimic)
# =============================================================================
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
except ImportError:
    print("\n[ERROR] 找不到 stable-baselines3。")
    print("请确保已执行：pip install gymnasium shimmy 'stable-baselines3[extra]'")
    sys.exit(1)

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.env_utils as EnvUtils

# 导入你写的 Wrapper
try:
    from wuhao.adaptive_env_utils_parallel import BatchedAdaptiveWrapper, RobomimicToGymWrapper, DSRLStylePolicy
    print("[INFO] 成功从 wuhao 模块导入 AdaptiveChunkingWrapper")
except ImportError:
    # 备选路径：如果脚本在 robomimic/scripts
    sys.path.append(os.path.abspath(os.path.join(current_dir, "../../")))
    try:
        from wuhao.adaptive_env_utils_parallel import BatchedAdaptiveWrapper, RobomimicToGymWrapper, DSRLStylePolicy
        print("[INFO] 成功通过备选路径导入 AdaptiveChunkingWrapper")
    except ImportError:
        print(f"[ERROR] 无法找到 wuhao.adaptive_env_utils。请检查文件是否存在。")
        sys.exit(1)



from typing import Callable

import json

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    线性学习率调度器。
    :param initial_value: 初始学习率。
    :return: 接受当前进度（1.0 到 0.0）并返回对应学习率的函数。
    """
    def func(progress_remaining: float) -> float:
        """
        progress_remaining 从 1.0 逐渐变为 0.0
        """
        return progress_remaining * initial_value
    return func




class SuccessBestModelCallback(BaseCallback):
    def __init__(self, check_freq, log_dir, ckpt_dir, window_size=50, verbose=1):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir
        self.best_success_rate = -1.0
        self.success_buffer = deque(maxlen=window_size)
        self.exec_lengths = []

        # --- 新增：专门记录动作（Chunk长度）的缓冲区 ---
        self.action_history = []
        
        # --- 新增：用于存储指标历史的字典 ---
        self.history = {
            "timesteps": [],
            "metrics": {}
        }
        self.json_path = os.path.join(log_dir, "training_metrics.json")

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # 记录执行长度
        for info in self.locals["infos"]:
            if "exec_len" in info:
                self.exec_lengths.append(info["exec_len"])

        # --- 【核心新增】：记录 RL 刚刚选下的动作 ---
        # actions 形状通常是 (n_envs,)
        actions = self.locals["actions"]
        self.action_history.extend(actions.flatten().tolist())
        
        # 统计成功率
        for i, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][i]
                raw_success = info.get("is_success", False)
                if isinstance(raw_success, dict):
                    is_success = raw_success.get("task", False)
                else:
                    is_success = bool(raw_success)
                self.success_buffer.append(1 if is_success else 0)
        return True

    def _on_rollout_end(self) -> None:
        # 1. 记录自定义指标到 TensorBoard
        
        if len(self.exec_lengths) > 0:
            avg_exec = np.mean(self.exec_lengths)
            self.logger.record("adaptive/avg_exec_len", avg_exec)
            self.exec_lengths = []

        # 2. 核心修复：遍历输出格式，找到 TensorBoard 句柄
        if len(self.action_history) > 0:
            for output_format in self.logger.output_formats:
                if isinstance(output_format, TensorBoardOutputFormat):
                    # 成功找到 TensorBoard 的 SummaryWriter
                    print("[INFO] 记录动作分布直方图到 TensorBoard...")
                    output_format.writer.add_histogram(
                        "adaptive/action_distribution", 
                        np.array(self.action_history), 
                        self.num_timesteps
                    )
            # 清空本次 Rollout 的动作缓存
            self.action_history = []
        
        if len(self.success_buffer) > 0:
            current_sr = np.mean(self.success_buffer)
            self.logger.record("adaptive/success_rate_recent", current_sr)
            
            if current_sr > self.best_success_rate:
                self.best_success_rate = current_sr
                path = os.path.join(self.ckpt_dir, "best_model.zip")
                self.model.save(path)
                # # 【核心修复】：同步保存 VecNormalize 统计量
                # # 我们需要从 venv 中找到 VecNormalize 层
                # if isinstance(self.training_env, VecNormalize):
                #     print("[INFO] 保存 VecNormalize 统计量...")
                #     self.training_env.save(os.path.join(self.ckpt_dir, "best_vec_normalize.pkl"))
                # else:
                #     print("[WARNING] 训练环境不是 VecNormalize，无法保存统计量。")
                if self.verbose > 0:
                    print(f"New Best Success Rate: {current_sr:.2f}! Model saved to {path}")

        # 2. --- 新增：捕获所有 Logger 指标并保存为 JSON ---
        self.history["timesteps"].append(self.num_timesteps)
        # 遍历当前 Logger 中所有的标量值 (包括 PPO 的 loss, reward 等)
        for key, value in self.logger.name_to_value.items():
            if key not in self.history["metrics"]:
                self.history["metrics"][key] = []
            self.history["metrics"][key].append(float(value))

        # 实时写入 JSON，防止崩溃丢失数据
        with open(self.json_path, "w") as f:
            json.dump(self.history, f, indent=4)
                    

def make_raw_env(env_meta, env_id, args, render_offscreen=False):
    import sys
    import os
    
    # 1. 路径修复
    robosuite_dir = "/home/wuhao/jobspace/robosuite"
    if robosuite_dir not in sys.path:
        sys.path.insert(0, robosuite_dir)

    # 2. 修复 robosuite
    import robosuite
    import robosuite.environments.base as suite_base
    if robosuite.__file__ is None:
        robosuite.__file__ = os.path.join(robosuite_dir, "robosuite/__init__.py")
    if not hasattr(robosuite, "make"):
        robosuite.make = suite_base.make
    if not hasattr(robosuite, "__version__"):
        robosuite.__version__ = "1.4.1"

    # 3. 手动注册 ToolHang
    try:
        from robosuite.environments.manipulation.tool_hang import ToolHang
        suite_base.register_env(ToolHang)
    except Exception:
        pass

    # 4. 【核心修复】：以正确的格式初始化子进程的 ObsUtils
    import robomimic.utils.obs_utils as ObsUtils
    if ObsUtils.OBS_KEYS_TO_MODALITIES is None:
        # 从 env_meta 获取所有观测键
        # 3090 运行环境下，我们通常只用 low_dim（传感器数据）
        all_keys = env_meta.get("all_obs_keys", [])
        if not all_keys:
            # 兜底：如果元数据里没有，手动指定 ToolHang 任务常见的键
            all_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
        
        # --- 格式修正：显式构造嵌套字典 ---
        # 第一级是模态名称，第二级是对应的键名列表
        obs_specs = {
            "obs": {
                "low_dim": all_keys,
                "rgb": []
            }
        }
        
        # 调用时，robomimic 往往只需要这个嵌套字典里的 "obs" 部分
        try:
            ObsUtils.initialize_obs_utils_with_obs_specs(obs_specs)
        except Exception:
            # 如果上面那个还报错，尝试更扁平的结构（适配不同 robomimic 版本）
            ObsUtils.initialize_obs_utils_with_obs_specs({"low_dim": all_keys})
        
        print(f"[SUB-PROCESS] ObsUtils 初始化成功，监控键: {len(all_keys)} 个")


    import robomimic.utils.env_utils as EnvUtils
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=render_offscreen, 
        use_image_obs=False
    )
    
    # 2. 包装成标准 Gym 接口
    env = RobomimicToGymWrapper(env)

    # 3. 传入视频配置
    video_dir = os.path.join(args.log_dir, "videos") if args.save_videos else None
    
    # 3. 【核心新增】：包装变长执行器 (运行在子进程)
    # 这里将 reward_offset 设为 1.0 (DSRL 风格)
    from wuhao.adaptive_env_utils_parallel import InternalVariableChunkWrapper
    env = InternalVariableChunkWrapper(
        env, 
        reward_offset=1.0, 
        max_steps=600,
        save_videos=args.save_videos,
        video_dir=video_dir,
        env_id=env_id
    )
    
    return env

def train(args):
    # --- A. 设备与策略加载 ---
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    print(f"[INFO] 使用设备: {device}")

    print(f"[INFO] 加载预训练策略: {args.agent}")
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=args.agent, 
        device=device, 
        verbose=True
    )
    
    # 彻底锁定预训练模型 (Freeze)
    # === 修正部分开始 ===
    # RolloutPolicy 内部包含真正的算法对象 policy.policy (即 DiffusionPolicyUNet)
    # 我们需要对内部的算法对象调用 eval() 和 requires_grad_(False)
    algo_instance = policy.policy 

    # 1. 强制解锁配置并对齐 Horizon
    # 获取模型训练时设定的预测长度 (通常是 16)
    pred_h = algo_instance.algo_config.horizon.prediction_horizon

    with algo_instance.algo_config.values_unlocked():
        # 将输出长度强行设为预测长度
        algo_instance.algo_config.horizon.action_horizon = pred_h
        print(f"[INFO] 已将 algo_config.action_horizon 从 {algo_instance.algo_config.horizon.action_horizon} 强制修改为 {algo_instance.algo_config.horizon.action_horizon}")

    # 2. 【最关键一步】：同步更新实例属性
    # Robomimic 的 DiffusionPolicy 在运行中通常直接引用 self.ac_horizon
    if hasattr(algo_instance, 'ac_horizon'):
        algo_instance.ac_horizon = pred_h
        print(f"[SUCCESS] 已强制覆盖 algo_instance.ac_horizon 为 {algo_instance.ac_horizon}")

    algo_instance.nets.eval() # 锁定网络中的 BatchNorm/Dropout
    for param in algo_instance.nets.parameters():
        param.requires_grad = False # 彻底锁定参数

    # 强制禁用 DiffusionPolicyUNet 内部原有的手动调度逻辑
    # 注意：这两个属性是在 algo_instance (DiffusionPolicyUNet) 上的
    algo_instance.use_action_scheduler = False 
    algo_instance.use_ood_monitor = False
    # === 修正部分结束 ===

    print("[INFO] 预训练模型已通过内部 algo 实例冻结")

    # --- B. 获取环境元数据 ---
    # 直接从之前加载好的 ckpt_dict 中获取环境元数据
    env_meta = ckpt_dict["env_metadata"]
    
    # --- 新增：强制设置最大执行步数为 600 ---
    env_meta["env_kwargs"]["horizon"] = 600
    print(f"[INFO] 强制设置环境最大步数 (Horizon) 为: {env_meta['env_kwargs']['horizon']}")
    
    # --- C. 定义环境工厂 (针对 SB3 向量化) ---
    # def make_env():
    #     # 1. 创建 Robomimic 原始环境
    #     raw_env = EnvUtils.create_env_from_metadata(
    #         env_meta=env_meta,
    #         render=False, 
    #         render_offscreen=False,
    #         use_image_obs=False, 
    #     )
        
    #     # 2. 包装自定义 RL 逻辑
    #     env = AdaptiveChunkingWrapper(
    #         env=raw_env,
    #         policy=policy,
    #         max_chunk_len=args.max_chunk_len,
    #         device=device,
    #         reward_offset=args.reward_offset,
    #         resample_penalty=args.resample_penalty
    #     )
        
    #     # 3. Monitor 用于记录奖励信息
    #     env = Monitor(env)
    #     return env

    # 1. 首先创建一个普通的 DummyVecEnv，里面只有原始的 Robomimic 环境
    # def make_raw_env():
    #     # 1. 创建 Robomimic 原始环境
    #     env = EnvUtils.create_env_from_metadata(
    #         env_meta=env_meta,
    #         render=False,
    #         render_offscreen=False,
    #         use_image_obs=False
    #     )
    #     # 2. 包装成标准 Gym 环境
    #     gym_env = RobomimicToGymWrapper(env)
        
    #     # 3. 核心修正：必须包装 Monitor 才能记录 ep_rew_mean 和 ep_len_mean
    #     # return Monitor(gym_env) 
    #     return gym_env 

    from functools import partial


    # 1. 设置视频文件夹路径到 logs 目录下
    video_dir = os.path.join(args.log_dir, "videos") if args.save_videos else None
    # 2. 这里的 offscreen 必须为 True 才能拿到画面
    offscreen = True if args.save_videos else False
    # env_fn = partial(make_raw_env, env_meta, render_offscreen=offscreen)
    env_fns = [partial(make_raw_env, env_meta, i, args, render_offscreen=offscreen) 
               for i in range(args.n_envs)]

    # --- 关键修改：Dummy -> Subproc ---
    # 使用 SubprocVecEnv，启动方式已经在 main 里设为 spawn 了
    raw_venv = SubprocVecEnv(env_fns)


    venv = BatchedAdaptiveWrapper(
        venv=raw_venv,
        policy=policy,
        max_chunk_len=args.max_chunk_len,
        device=device,
        reward_offset=args.reward_offset,
        resample_penalty=args.resample_penalty,
    )

    # 4. 【核心修复】：在最外层包装 VecMonitor
    # 这样 VecMonitor 就能看到你 Wrapper 返回的每一个 done=True 和 reward
    venv = VecMonitor(venv)
    # venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_obs=10.)



    lr_schedule = linear_schedule(args.lr)

    # 3. 在 PPO 中使用配置
    # DSRL 配置：3 层 2048，Tanh 激活
    dsrl_policy_kwargs = dict(
        net_arch=dict(pi=[2048, 2048, 2048], vf=[2048, 2048, 2048]), 
        activation_fn=nn.Tanh,
        ortho_init=False, # DSRL 似乎没有显式强调 ortho_init，默认即可，但 LayerNorm 对初始化不那么敏感
        # log_std_init=0.0 # PPO 默认就是 0.0
    )

    # --- E. 配置 PPO 模型 ---
    model = PPO(
        DSRLStylePolicy,
        venv,
        policy_kwargs=dsrl_policy_kwargs,
        verbose=1,
        learning_rate=lr_schedule,  # 【修改点】：传入函数而非固定值
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=0.95,
        ent_coef=args.ent_coef, 
        # tensorboard_log=os.path.join(args.output, "tb_logs"),
        device=device,
        tensorboard_log=args.log_dir
    )

    actual_save_freq = max(args.save_freq // args.n_envs, 1)

    # --- F. 回调与训练 ---
    checkpoint_callback = CheckpointCallback(
        save_freq=actual_save_freq,
        save_path=args.ckpt_dir,  # 修改为使用 args.ckpt_dir
        # save_path=os.path.join(args.output, "checkpoints"),
        name_prefix="adaptive_agent"
    )

    

    adaptive_cb = SuccessBestModelCallback(
        check_freq=args.n_steps, # 虽然没用到，但保持与采样步数一致较好
        log_dir=args.log_dir,
        ckpt_dir=args.ckpt_dir,
        window_size=args.window_size # 从 args 传入
    )

    print(f"[INFO] 开始 RL 训练。总步数: {args.total_timesteps}")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, adaptive_cb], # 加上自定义回调
        progress_bar=True
    )

    # --- G. 保存最终结果 ---
    final_path = os.path.join(args.output, "adaptive_final_model.zip")
    model.save(final_path)
    print(f"[SUCCESS] 训练完成！模型保存至: {final_path}")
    venv.close()



if __name__ == "__main__":
    # 3. 将 'fork' 改为 'spawn'
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--agent", type=str, required=True, help="预训练模型 (.pth) 路径")
    parser.add_argument("--output", type=str, default="./adaptive_rl_results", help="输出目录")
    
    parser.add_argument("--max_chunk_len", type=int, default=16, help="Action Chunk 最大长度")
    parser.add_argument("--resample_penalty", type=float, default=-0.05, help="重采样惩罚")
    parser.add_argument("--reward_offset", type=float, default=0.0, help="奖励偏移量")
    
    parser.add_argument("--total_timesteps", type=int, default=1000000)
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--save_freq", type=int, default=20000)

    # 新增：自定义目录参数
    # 如果不传，默认用时间戳命名，防止覆盖
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--log_dir", type=str, default=f"logs/run_{timestamp}")
    parser.add_argument("--ckpt_dir", type=str, default=f"checkpoints/run_{timestamp}")

    # 视频与 JSON 保存开关
    parser.add_argument("--window_size", type=int, default=50, help="计算平均成功率的窗口大小")
    parser.add_argument("--save_videos", action="store_true", help="是否保存视频")
    parser.add_argument("--video_freq", type=int, default=10000, help="多少步录制一个视频")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    train(args)