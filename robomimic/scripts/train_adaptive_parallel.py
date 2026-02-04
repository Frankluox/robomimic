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
import random


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
# repo_root = os.path.abspath(os.path.join(current_dir, "../../..")) 
repo_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(repo_root)

# =============================================================================
# 2. 依赖导入 (SB3 & Robomimic)
# =============================================================================
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.buffers import RolloutBuffer
    from stable_baselines3.common.utils import obs_as_tensor, safe_mean
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



# =============================================================================
# 1. Core Implementation of SMDP Components
# =============================================================================

class SMDPRolloutBuffer(RolloutBuffer):
    """
    支持变长步数 (Duration) 的 Rollout Buffer
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 存储每一步的物理时长 k
        self.durations = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def reset(self) -> None:
        # 必须重置 durations，防止脏数据
        self.durations = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        super().reset()

    # --- [Fix 1] 显式重写 add，避免 **kwargs 传递给不支持它的父类 ---
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        durations: np.ndarray = None, # 新增参数
    ) -> None:
        
        # 1. 存储 duration
        if durations is not None:
            self.durations[self.pos] = np.array(durations).copy()
        
        # 2. 调用父类 add (不带 durations 参数)
        super().add(obs, action, reward, episode_start, value, log_prob)

    def compute_returns_and_advantage(self, last_values, dones):
        """
        Hardcore SMDP GAE Calculation:
        Discount is gamma^k instead of gamma
        """
        last_values = last_values.clone().cpu().numpy().flatten()
        last_gae_lam = 0

        # [探针-GAE] 我们只打印第一个环境的最后 5 步计算过程，用于人工验算
        debug_env_idx = 0 
        debug_steps = []

        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]

            # --- [SMDP Core] Dynamic Discounting ---
            # k = 0 (resample) -> gamma^0 = 1.0 (No discount for immediate retry)
            # k > 0 -> gamma^k (Physical discount)
            current_k = self.durations[step]

            # 这里的 self.gamma 是基础 gamma (比如 0.99)
            dynamic_gamma = self.gamma ** current_k
            
            # GAE Formula with dynamic gamma
            # delta = r + gamma^k * V(s') - V(s)
            delta = self.rewards[step] + dynamic_gamma * next_values * next_non_terminal - self.values[step]
            
           # Advantage = delta + (gamma * lambda)^k * next_advantage
            # 我们需要把 (gamma * lambda) 作为一个整体进行 k 次幂衰减
            discount_for_advantage = (self.gamma * self.gae_lambda) ** current_k

            last_gae_lam = delta + discount_for_advantage * next_non_terminal * last_gae_lam
            
            self.advantages[step] = last_gae_lam

            # [探针-GAE] 记录倒数几步的数据
            if step >= self.buffer_size - 5: # 只看最后 5 步
                # 提取第 0 个环境的数据来打印
                k_val = current_k[debug_env_idx]
                r_val = self.rewards[step][debug_env_idx]
                v_curr = self.values[step][debug_env_idx]
                v_next = next_values[debug_env_idx] if isinstance(next_values, np.ndarray) else next_values
                non_term = next_non_terminal[debug_env_idx] if isinstance(next_non_terminal, np.ndarray) else next_non_terminal
                
                debug_steps.append({
                    "step": step,
                    "k": k_val,
                    "r": r_val,
                    "v_t": v_curr,
                    "v_t+1": v_next,
                    "gamma^k": dynamic_gamma[debug_env_idx],
                    "delta": delta[debug_env_idx],
                    "adv": last_gae_lam[debug_env_idx]
                })
        
        self.returns = self.advantages + self.values

        # # [探针-GAE] 打印验算表
        # print("\n" + "="*80)
        # print("[PROBE-GAE] Verification Table (Last 5 Steps of Env 0)")
        # print(f"{'Step':<6} | {'k':<4} | {'Reward':<10} | {'V(t)':<10} | {'V(t+1)':<10} | {'Gam^k':<8} | {'Delta':<10} | {'Advantage':<10}")
        # print("-" * 80)
        # for d in debug_steps: # 注意 debug_steps 是倒序存的 (last step first)
        #     print(f"{d['step']:<6} | {d['k']:<4.0f} | {d['r']:<10.4f} | {d['v_t']:<10.4f} | {d['v_t+1']:<10.4f} | {d['gamma^k']:<8.4f} | {d['delta']:<10.4f} | {d['adv']:<10.4f}")
        # print("="*80 + "\n")

class SMDPPPO(PPO):
    """
    Custom PPO that collects 'exec_len' (k) and uses SMDPRolloutBuffer
    """
    def __init__(self, *args, **kwargs):
        # Force usage of our custom buffer
        kwargs["rollout_buffer_class"] = SMDPRolloutBuffer
        super().__init__(*args, **kwargs)
        self.debug_print_once = False # 用于控制打印频率

    def collect_rollouts(
        self,
        env,
        callback,
        rollout_buffer,
        n_rollout_steps,
    ) -> bool:
        """
        Overridden to capture 'exec_len' from infoAND fix timeout bootstrapping
        """
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        # --- 【新增】：为了调用工具方法，先从封装层里找到真正的 BatchedAdaptiveWrapper ---
        curr_env = env
        while not isinstance(curr_env, BatchedAdaptiveWrapper):
            if hasattr(curr_env, "venv"):
                curr_env = curr_env.venv
            else:
                break
        real_adaptive_env = curr_env
        # ------------------------------------------------------------------------

        n_steps = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with torch.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()
            clipped_actions = actions # Discrete action space, no clipping needed usually

            # Execute Step
            new_obs, rewards, dones, infos = env.step(clipped_actions)

            # --- [SMDP Core] Extract Durations ---
            # We need to extract 'exec_len' (k) from infos
            durations = np.zeros(env.num_envs, dtype=np.float32)
            for i, info in enumerate(infos):
                # If chunk k=5, exec_len=5. If resample k=0, exec_len=0.
                # durations[i] = info.get("exec_len", 1.0) # Default to 1 if missing
                durations[i] = info.get("exec_len")
            # -------------------------------------

            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            # Reshape actions
            if isinstance(self.action_space, torch.nn.Module): # Wait, checking space type
                pass # PPO handles this usually

            # --- [Fix 2] Handle timeout bootstrapping with SMDP Discount ---
            for idx, done in enumerate(dones):
                if done:
                    # [超级探针]：只要 Done 了，就把 info 打印出来看看！
                    # 只打印一次，防止日志爆炸
                    # if not self.debug_print_once:
                    #     print(f"\n[DEBUG PROBE] Env {idx} DONE triggered!")
                    #     print(f"  Keys in info: {list(infos[idx].keys())}")
                    #     print(f"  Has 'terminal_observation'?: {'terminal_observation' in infos[idx]}")
                    #     print(f"  Has 'TimeLimit.truncated'?: {infos[idx].get('TimeLimit.truncated')}")
                    #     print(f"  Raw Info: {infos[idx]}")
                    #     self.debug_print_once = True

                    # 只有同时满足这两个条件，才是超时截断
                    if (
                        infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                    ):
                        # =====================================================
                        # 【核心修正】：使用你写好的 get_terminal_rl_obs
                        # =====================================================
                        raw_terminal_obs_dict = infos[idx]["terminal_observation"]
                        
                        # 1. 调用工具方法：字典 -> 277维向量
                        flat_obs = real_adaptive_env.get_terminal_rl_obs(raw_terminal_obs_dict, env_idx=idx)
                        
                        # 2. 转换为 Tensor 并增加 Batch 维度
                        terminal_obs_tensor = obs_as_tensor(flat_obs, self.device).unsqueeze(0)
                        
                        with torch.no_grad():
                            # 3. 现在 Policy 认得这个向量了，可以预测价值
                            terminal_value = self.policy.predict_values(terminal_obs_tensor)[0]
                        # =====================================================

                        # print(f"[PROBE-PPO] Timeout Truncation CONFIRMED!")
                        # SMDP 修正：补全 V 值
                        rewards[idx] += (self.gamma ** durations[idx]) * terminal_value
                        # print(f"  Applied Bootstrap: {((self.gamma ** durations[idx]) * terminal_value).item():.4f}")
            
            # Add to buffer with durations

            # [探针-Buffer存入] 随机抽查存入的数据
            # if n_steps % 100 == 0: # 每100步抽查一次
            #      print(f"[PROBE-PPO] Adding to buffer: k={durations[0]}, r={rewards[0]:.4f}")

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                durations=durations # <--- Pass k here
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with torch.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.on_rollout_end()
        return True


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
    def __init__(self, check_freq, log_dir, ckpt_dir, window_size=50, verbose=1, max_chunk_len=32):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir
        self.max_chunk_len = max_chunk_len # <--- 新增：记录最大动作值
        self.best_success_rate = -1.0
        self.success_buffer = deque(maxlen=window_size)
        self.exec_lengths = []

        # --- 新增：专门记录动作（Chunk长度）的缓冲区 ---
        self.action_history = []

        # --- 【新增】：用于存储观测值统计信息的缓冲区 ---
        self.obs_vec_averages = [] 
        # --------------------------------------------
        
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

        # --- 【核心新增】：捕获观测值并计算浓缩平均值 ---
        # new_obs 的形状通常是 (n_envs, 277)
        if "new_obs" in self.locals:
            obs = self.locals["new_obs"]
            # 对每个向量（axis=1）算平均值，得到 (n_envs,) 的向量
            # 这个向量代表了当前 Batch 中每个环境观测的“浓缩值”
            batch_obs_averages = np.mean(obs, axis=1)
            self.obs_vec_averages.extend(batch_obs_averages.tolist())
        else:
            print("[WARNING] 在回调中找不到 'new_obs'，无法记录观测值统计信息。")
        # --------------------------------------------
        
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

        # --- 【核心新增】：计算并记录观测值的均值和标准差 ---
        if len(self.obs_vec_averages) > 0:
            # 这些浓缩平均值的均值
            m_of_m = np.mean(self.obs_vec_averages)
            # 这些浓缩平均值的标准差
            s_of_m = np.std(self.obs_vec_averages)
            
            self.logger.record("adaptive/obs_condensed_mean", m_of_m)
            self.logger.record("adaptive/obs_condensed_std", s_of_m)
            
            # 清空缓存用于下次 Rollout
            self.obs_vec_averages = []
        else:
            print("[WARNING] 本次 Rollout 中没有观测值统计信息，无法记录均值和标准差。")
        # -----------------------------------------------

        # 2. 核心修复：遍历输出格式，找到 TensorBoard 句柄
        if len(self.action_history) > 0:
            actions_np = np.array(self.action_history)
            # 使用 bincount 计算 0 到 max_chunk_len 每个数字出现的次数
            counts = np.bincount(actions_np.astype(int), minlength=self.max_chunk_len + 1)
            # 转换为百分比（频率）
            distribution = (counts / len(self.action_history)).tolist()
            
            # 手动存入 history 字典，不经过 self.logger (因为 logger 只收标量)
            key = "adaptive/action_distribution_ratio"
            if key not in self.history["metrics"]:
                self.history["metrics"][key] = []
            self.history["metrics"][key].append(distribution)

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
        
        current_sr = np.mean(self.success_buffer) if len(self.success_buffer) > 0 else 0.0
        self.logger.record("adaptive/success_rate_recent", current_sr)

        # 5. 断点保存逻辑 (Best & Latest)
        is_new_best = current_sr > self.best_success_rate
        save_names = ["latest_model"]

        if is_new_best:
            self.best_success_rate = current_sr
            save_names.append("best_model")

        for name in save_names:
            path = os.path.join(self.ckpt_dir, name)
            self.model.save(path)
            # if isinstance(self.training_env, VecNormalize):
            #     self.training_env.save(f"{path}_vec_normalize.pkl")
            # 保存回调状态
            cb_state = {"best_success_rate": self.best_success_rate, "success_buffer": list(self.success_buffer)}
            with open(f"{path}_cb_state.json", "w") as f: json.dump(cb_state, f)

        # --- 2. 获取 SB3 自动记录的 Reward ---
        # VecMonitor 会把 ep_rew_mean 存入 logger 的 name_to_value 字典
        ep_rew_mean = self.logger.name_to_value.get("rollout/ep_rew_mean", -np.inf)

        # --- 3. 控制台打印 (Log 展示) ---
        if self.verbose > 0:
            print("-" * 30)
            print(f"Step: {self.num_timesteps}")
            print(f"Recent Success Rate: {current_sr:.4f}")
            if ep_rew_mean != -np.inf:
                print(f"Mean Episode Reward: {ep_rew_mean:.2f}")
            else:
                print("Mean Episode Reward: N/A (Waiting for first episode done)")
            print("-" * 30)

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

    def load_resume_state(self, resume_cb_path, resume_json_path):
        """从文件恢复回调状态和 JSON 历史"""
        if os.path.exists(resume_cb_path):
            with open(resume_cb_path, "r") as f:
                state = json.load(f)
                self.best_success_rate = state["best_success_rate"]
                self.success_buffer = deque(state["success_buffer"], maxlen=self.window_size)
            print(f"[RESUME] Callback 状态已加载，当前最佳SR: {self.best_success_rate:.4f}")
        if os.path.exists(resume_json_path):
            with open(resume_json_path, "r") as f:
                self.history = json.load(f)
            print(f"[RESUME] 历史指标 JSON 已加载。")
                    

def make_raw_env(env_meta, env_id, args, render_offscreen=False):
    import sys
    import os

    # --- 【新增：设置唯一随机种子】 ---
    # 结合基础种子 (args.seed) 和环境 ID (env_id)
    base_seed = getattr(args, "seed", 42)
    worker_seed = base_seed + env_id
    
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(worker_seed)
    
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
    # video_dir = os.path.join(args.log_dir, "videos") if args.save_videos else None
    video_dir = args.video_dir if args.save_videos else None
    
    # 3. 【核心新增】：包装变长执行器 (运行在子进程)
    # 这里将 reward_offset 设为 1.0 (DSRL 风格)
    from wuhao.adaptive_env_utils_parallel import InternalVariableChunkWrapper
    env = InternalVariableChunkWrapper(
        env, 
        reward_offset=1.0, 
        max_steps=600,
        save_videos=args.save_videos,
        video_dir=video_dir,
        env_id=env_id,
        gamma=args.gamma  # <--- Pass gamma here
    )
    
    return env

def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # --- 1. 统一路径管理与续训逻辑 ---
    if args.resume:
        # 如果指定了 resume，直接指向该文件夹
        full_run_dir = args.resume
        if not os.path.exists(full_run_dir):
            raise FileNotFoundError(f"指定的续训目录不存在: {full_run_dir}")
        print(f"[RESUME] 检测到续训请求，将使用现有目录: {full_run_dir}")
    else:
        # 如果没有指定 resume，则创建新的带时间戳的文件夹
        run_name = f"PPO_{datetime.now().strftime('%m%d_%H%M%S')}"
        full_run_dir = os.path.join(args.log_dir, run_name)
        print(f"[NEW] 开始新训练，目录: {full_run_dir}")


    # 更新各个保存路径
    args.ckpt_dir = os.path.join(full_run_dir, "checkpoints")
    args.video_dir = os.path.join(full_run_dir, "videos")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.video_dir, exist_ok=True)
    os.makedirs(full_run_dir, exist_ok=True)

    # --- 2. 断点检测逻辑 ---
    # 检查目录下是否存在最新模型
    resume_ckpt = os.path.join(args.ckpt_dir, "latest_model.zip")
    resume_cb = os.path.join(args.ckpt_dir, "latest_model_cb_state.json")
    resume_json = os.path.join(full_run_dir, "training_metrics.json")

    # 只有当文件确实存在时，才标记为 is_resuming
    is_resuming = os.path.exists(resume_ckpt)
    if is_resuming:
        print(f"[INFO] 找到断点文件: {resume_ckpt}，准备恢复...")
    elif args.resume:
        print(f"[WARNING] 虽然指定了 resume 目录，但未找到 latest_model.zip，将从头开始训练。")

    # --- 3. 自动打印/保存 Config (仅限新训练) ---
    if not is_resuming:
        config_path = os.path.join(full_run_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(vars(args), f, indent=4)
        print(f"[INFO] 训练配置已保存至: {config_path}")


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
        action_history_len=args.action_history_len # <--- 传入
    )

    # 4. 【核心修复】：在最外层包装 VecMonitor
    # 这样 VecMonitor 就能看到你 Wrapper 返回的每一个 done=True 和 reward
    venv = VecMonitor(venv)
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_obs=10.)
    # venv = VecNormalize(venv, norm_obs=False, norm_reward=True)\


    # --- D. 模型加载/初始化 ---
    if is_resuming:
        print(f"[RESUME] 从断点加载模型: {resume_ckpt}")
        model = SMDPPPO.load(resume_ckpt, env=venv, custom_objects={"learning_rate": linear_schedule(args.lr)})
    else:
        # 在 PPO 中使用配置
        # DSRL 配置：3 层 2048，Tanh 激活
        dsrl_policy_kwargs = dict(
            net_arch=dict(pi=[2048, 2048, 2048], vf=[2048, 2048, 2048]), 
            activation_fn=nn.Tanh,
            ortho_init=False, # DSRL 似乎没有显式强调 ortho_init，默认即可，但 LayerNorm 对初始化不那么敏感
            # log_std_init=0.0 # PPO 默认就是 0.0
        )
        # --- E. 配置 PPO 模型 ---
        model = SMDPPPO(
            DSRLStylePolicy,
            venv,
            policy_kwargs=dsrl_policy_kwargs,
            verbose=1,
            learning_rate=linear_schedule(args.lr),  # 【修改点】：传入函数而非固定值
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef, 
            # tensorboard_log=os.path.join(args.output, "tb_logs"),
            device=device,
            tensorboard_log=full_run_dir,
            # tb_log_name=run_name
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
        log_dir=full_run_dir,
        ckpt_dir=args.ckpt_dir,
        window_size=args.window_size, # 从 args 传入
        max_chunk_len=args.max_chunk_len, # <--- 必须传入此参数
    )

    if is_resuming:
        adaptive_cb.load_resume_state(resume_cb, resume_json)

    print(f"[INFO] 开始 RL 训练。总步数: {args.total_timesteps}")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, adaptive_cb], # 加上自定义回调
        progress_bar=True,
        # 修复点：tb_log_name 应该在这里！
        # 设为 "scalars" 后，日志会存在 full_run_dir/scalars_1/
        tb_log_name="scalars"
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
    parser.add_argument("--gae_lambda", type=float, default=0.95)
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

    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--action_history_len", type=int, default=0, help="动作历史窗口长度（0为关闭）")

    parser.add_argument("--resume", type=str, default=None, help="指定要续训的文件夹路径 (例如: logs/run_xxx/PPO_0129_000156)")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    train(args)