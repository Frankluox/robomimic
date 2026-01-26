"""
整合版 train_adaptive.py
功能：加载冻结的预训练 Diffusion Policy，通过 PPO 训练 RL Agent 动态选取 Action Chunk 长度。
"""




import multiprocessing
multiprocessing.set_start_method('fork', force=True)

import argparse
import sys
import os
import torch
import numpy as np
from collections import deque
from datetime import datetime

from stable_baselines3.common.monitor import Monitor

# --- 新增：强制修复 robosuite 版本属性缺失问题 ---
try:
    import robosuite
    if not hasattr(robosuite, "__version__"):
        # 赋予一个默认的 v1 版本号，让 robomimic 能够继续执行
        robosuite.__version__ = "1.4.1" 
        print("[INFO] 手动为 robosuite 模块添加了 __version__ 属性")
except ImportError:
    pass
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
    from wuhao.adaptive_env_utils import AdaptiveChunkingWrapper, BatchedAdaptiveWrapper, RobomimicToGymWrapper
    print("[INFO] 成功从 wuhao 模块导入 AdaptiveChunkingWrapper")
except ImportError:
    # 备选路径：如果脚本在 robomimic/scripts
    sys.path.append(os.path.abspath(os.path.join(current_dir, "../../")))
    try:
        from wuhao.adaptive_env_utils import AdaptiveChunkingWrapper, BatchedAdaptiveWrapper, RobomimicToGymWrapper
        print("[INFO] 成功通过备选路径导入 AdaptiveChunkingWrapper")
    except ImportError:
        print(f"[ERROR] 无法找到 wuhao.adaptive_env_utils。请检查文件是否存在。")
        sys.exit(1)





class SuccessBestModelCallback(BaseCallback):
    def __init__(self, check_freq, log_dir, ckpt_dir, window_size=50, verbose=1):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir
        self.best_success_rate = -1.0
        self.success_buffer = deque(maxlen=window_size)
        self.exec_lengths = []

        # 确保目录存在
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # 1. 记录执行长度
        for info in self.locals["infos"]:
            if "exec_len" in info:
                self.exec_lengths.append(info["exec_len"])
        
        # 2. 统计成功率 (关键：从 info 获取 robosuite 的 success 标志)
        for i, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][i]
                # 兼容性处理：优先找 info 里的 success，否则看 reward
                is_success = info.get("success", self.locals["rewards"][i] > 0.5)
                self.success_buffer.append(1 if is_success else 0)
        return True

    def _on_rollout_end(self) -> None:
        # 打印并记录到 TensorBoard
        if len(self.exec_lengths) > 0:
            self.logger.record("adaptive/avg_exec_len", np.mean(self.exec_lengths))
            self.exec_lengths = []
        
        if len(self.success_buffer) > 0:
            current_sr = np.mean(self.success_buffer)
            self.logger.record("adaptive/success_rate_recent", current_sr)
            
            # 如果成功率刷新纪录，保存 Best 模型
            if current_sr > self.best_success_rate:
                self.best_success_rate = current_sr
                path = os.path.join(self.ckpt_dir, "best_model.zip")
                self.model.save(path)
                if self.verbose > 0:
                    print(f"New Best Success Rate: {current_sr:.2f}! Model saved to {path}")
                    
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
    def make_raw_env():
        # 1. 创建 Robomimic 原始环境
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False,
            render_offscreen=False,
            use_image_obs=False
        )
        # 2. 包装成标准 Gym 环境
        gym_env = RobomimicToGymWrapper(env)
        
        # 3. 核心修正：必须包装 Monitor 才能记录 ep_rew_mean 和 ep_len_mean
        return Monitor(gym_env) 

    # 然后再创建向量环境
    raw_venv = DummyVecEnv([make_raw_env for _ in range(args.n_envs)])

    venv = BatchedAdaptiveWrapper(
        venv=raw_venv,
        policy=policy,
        max_chunk_len=args.max_chunk_len,
        device=device,
        reward_offset=args.reward_offset,
        resample_penalty=args.resample_penalty
    )

    # --- E. 配置 PPO 模型 ---
    model = PPO(
        "MlpPolicy",
        venv,
        verbose=1,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=args.gamma,
        gae_lambda=0.95,
        ent_coef=args.ent_coef, 
        # tensorboard_log=os.path.join(args.output, "tb_logs"),
        device=device,
        tensorboard_log=args.log_dir
    )

    # --- F. 回调与训练 ---
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=os.path.join(args.output, "checkpoints"),
        name_prefix="adaptive_agent"
    )

    adaptive_cb = SuccessBestModelCallback(
        check_freq=2048,
        log_dir=args.log_dir,
        ckpt_dir=args.ckpt_dir)

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
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--save_freq", type=int, default=20000)

    # 新增：自定义目录参数
    # 如果不传，默认用时间戳命名，防止覆盖
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--log_dir", type=str, default=f"logs/run_{timestamp}")
    parser.add_argument("--ckpt_dir", type=str, default=f"checkpoints/run_{timestamp}")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    train(args)