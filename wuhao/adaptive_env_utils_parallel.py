try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces


# import gym
import numpy as np
import torch
# from gymnasium import spaces # 注意：根据你的环境可能需要换回 gym
import robomimic.utils.tensor_utils as TensorUtils



from stable_baselines3.common.vec_env import VecEnv

import imageio
import os
import torch.nn as nn


class RobomimicToGymWrapper(gym.Env):
    def __init__(self, env):
        self.env = env
        self.action_space = spaces.Box(low=-1, high=1, shape=(env.action_dimension,), dtype=np.float32)
        obs = self.env.reset()
        space_dict = {k: spaces.Box(low=-np.inf, high=np.inf, shape=v.shape, dtype=v.dtype) for k, v in obs.items()}
        self.observation_space = spaces.Dict(space_dict)

    def reset(self, seed=None, options=None):
        return self.env.reset(), {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, False, info

    # 修改后 (增加默认分辨率 512x512)
    def render(self, mode="rgb_array", height=512, width=512): 
        # Robomimic 环境的 render 接受 height 和 width 参数
        return self.env.render(mode=mode, height=height, width=width)
# =============================================================================
# 新增：运行在子进程中的内部包装器
# 功能：接收 Chunk 和 k，自主循环执行，执行完后物理时钟自动停止
# =============================================================================
class InternalVariableChunkWrapper(gym.Wrapper):
    def __init__(self, env, reward_offset=0.0, max_steps=600, save_videos=False, video_dir=None, env_id=0):
        super().__init__(env)
        self.reward_offset = reward_offset
        self.max_steps = max_steps
        self.save_videos = save_videos
        self.video_dir = video_dir
        self.env_id = env_id  # 用于区分不同进程的文件名

        self.current_step = 0
        self.episode_count = 0
        self.episode_frames = []
        self.current_obs = None
        self.success_flag = False

        if self.save_videos and self.video_dir:
            os.makedirs(self.video_dir, exist_ok=True)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_step = 0 # 重置计数器
        self.episode_frames = []
        self.success_flag = False
        self.current_obs = obs
        return obs, info

    def step(self, action_data):
        chunk = action_data['chunk']
        k = int(action_data['k'])

        # 探针：记录执行前的底层物理步数 (针对 Robosuite)
        base_env = self.env.env
        step_before = base_env.cur_time if hasattr(base_env, 'cur_time') else 0
        
        if k <= 0:
            # 修改点：确保 resampled 时也返回 info 字典，且 success_flag 状态保持
            info = {"exec_len": 0, "resampled": True, "is_success": self.success_flag}
            return self.current_obs, 0.0, False, False, info

        total_reward = 0.0
        done = False
        actual_steps = 0
        
        for i in range(k):
            obs, reward, terminated, tr, info = self.env.step(chunk[i])
            self.current_step += 1 # 物理步数累加
            actual_steps += 1
            
            total_reward += (reward - self.reward_offset)
            self.current_obs = obs

            # --- 录制逻辑 ---
            if self.save_videos:
                # 调用包装好的 RobomimicToGymWrapper.render
                frame = self.env.render(mode="rgb_array", height=160, width=160)
                self.episode_frames.append(frame)

            # 判定成功（Robomimic 典型判定）
            if reward >= 1.0 or (isinstance(info.get('is_success'), dict) and info['is_success'].get('task')):
                self.success_flag = True

            
            
            
            # --- 核心逻辑：双重终止判定 ---
            # 1. 环境原生结束 (terminated/tr)
            # 2. 达到你设定的 600 步上限
            if terminated or tr or self.current_step >= self.max_steps or self.success_flag:
                done = True
                break


        # 探针：记录执行后的物理步数
        step_after = base_env.cur_time if hasattr(base_env, 'cur_time') else 0
        actual_physics_steps = actual_steps # 循环里的计数
        
        # 验证：物理步数增长必须等于实际循环次数
        # 如果物理步数增长大于 k，说明有隐形步进
        # print(f"[探针-物理时钟] 环境ID: {os.getpid()} | 请求k: {k} | 实际步进: {actual_physics_steps} | 物理时间增量: {step_after - step_before}")
        # if self.success_flag:
            # print(f"[探针-物理时钟] 环境ID: {os.getpid()} | 请求k: {k} | 实际步进: {actual_physics_steps} | 物理时间增量: {step_after - step_before}")
        

        # --- 核心修正：显式写回成功信号到 info ---
        # 这样主进程的 SuccessBestModelCallback 才能看到它
        info["is_success"] = self.success_flag

        info["exec_len"] = actual_steps
        info["resampled"] = False

        if done:
            # 探针：记录 done 瞬间的末尾坐标
            pos_at_done = self.current_obs.get('robot0_eef_pos', np.zeros(3)).copy()
            
            # 子进程会自动触发底层重置 (如果是 VecEnv 包装的话)
            # 或者如果是手动重置，在此观察
            # print(f"[探针-重置] 环境 {os.getpid()} 触发 Done. 终止坐标: {pos_at_done}")

            if self.save_videos and len(self.episode_frames) > 0:
                self._save_video()
            self.episode_count += 1
        
        return self.current_obs, total_reward, done, False, info

    def _save_video(self):
        import imageio
        status = "success" if self.success_flag else "fail"
        filename = f"env_{self.env_id}_ep_{self.episode_count:03d}_{status}.mp4"
        path = os.path.join(self.video_dir, filename)
        try:
            imageio.mimsave(path, self.episode_frames, fps=20)
        except Exception as e:
            print(f"[ERROR] 子进程 {self.env_id} 保存视频失败: {e}")
        self.episode_frames = [] # 释放内存


# =============================================================================
# 修改后的主进程包装器
# =============================================================================
class BatchedAdaptiveWrapper(VecEnv):
    def __init__(self, venv, policy, max_chunk_len, device, reward_offset=0.0, resample_penalty=-0.1, save_all_videos=False, video_dir=None):
        self.venv = venv
        self.num_envs = venv.num_envs
        self.policy = policy
        self.max_chunk_len = max_chunk_len
        self.device = device
        self.resample_penalty = resample_penalty
        
        self.algo_instance = policy.policy if hasattr(policy, 'policy') else policy
        self.action_dim = self.algo_instance.ac_dim
        
        rl_action_space = spaces.Discrete(max_chunk_len + 1)
        
        # 统计数据加载（用于归一化）
        self.stats = np.load("/home/wuhao/jobspace/robomimic/datasets/tool_hang/ph/toolhang_stats.npz")
        
        # 探测观测空间维度
        tmp_obs_batch = venv.reset()
        sample_obs_dict = tmp_obs_batch[0] if isinstance(tmp_obs_batch, list) else {k:v[0] for k,v in tmp_obs_batch.items()}
        obs_vec = self._get_obs_vec(sample_obs_dict)
        self.total_obs_dim = obs_vec.shape[0] + (max_chunk_len * self.action_dim)
        
        rl_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.total_obs_dim,), dtype=np.float32)
        super().__init__(self.num_envs, rl_observation_space, rl_action_space)

        self.current_obs_dicts = [None] * self.num_envs
        self.actions_cache = None

    def _get_obs_vec(self, obs_dict):
        # 静态归一化逻辑
        key_shapes = {'robot0_eef_pos':(3,), 'robot0_eef_quat':(4,), 'robot0_gripper_qpos':(2,), 'object':(44,)}
        vecs = []
        for k, shape in key_shapes.items():
            if k in obs_dict:
                val = obs_dict[k].flatten().astype(np.float32)
                val = (val - self.stats[f"{k}_mean"]) / self.stats[f"{k}_std"]
                vecs.append(val)
            else:
                vecs.append(np.zeros(shape, dtype=np.float32))
        
        final_vec = np.concatenate(vecs)
        
        # 探针：检查归一化后的量级
        if np.max(np.abs(final_vec)) > 20.0:
            key_name = "unknown" # 可以进一步定位是哪个 key 没归一化好
            # print(f"[探针-异常警告] 归一化后的特征量级过大 (>20)! 检查统计量对齐。Max: {np.max(final_vec):.2f}")
            
        return final_vec

    def _make_rl_obs(self, obs_dict, chunk):
        obs_vec = self._get_obs_vec(obs_dict)
        action_mean = self.stats["action_mean"]
        action_std = self.stats["action_std"]
        # 对整个 chunk 进行归一化
        normalized_chunk = (chunk - action_mean) / action_std
        if normalized_chunk.shape[0] < self.max_chunk_len:
            pad = np.zeros((self.max_chunk_len - normalized_chunk.shape[0], self.action_dim))
            normalized_chunk = np.concatenate([normalized_chunk, pad], axis=0)
        return np.concatenate([obs_vec, normalized_chunk.flatten()]).astype(np.float32)

    def step_async(self, actions):
        requested_ks = actions.astype(int)
        # 主进程在 GPU 上批量计算 Chunk
        batch_chunks = self._get_batch_policy_plan(self.current_obs_dicts)
        
        # 探针：观察 Batch 中的 k 分布
        # print(f"[探针-决策] 本轮 Batch 长度分配: {requested_ks.tolist()} | Max k: {np.max(requested_ks)}")
    
        # 构建指令包
        payloads = []
        for i in range(self.num_envs):
            payloads.append({
                'chunk': batch_chunks[i],
                'k': requested_ks[i]
            })
        
        self.actions_cache = requested_ks
        # 发送给子进程：每个进程现在会自己跑 k 步
        self.venv.step_async(payloads)

    def step_wait(self):
        # 1. 等待子进程返回各自执行完后的 Batch 结果
        obs_batch, rews_all, dones_all, infos_all = self.venv.step_wait()

        # 探针：检查是否有环境因为达到 600 步而终止
        for i, info in enumerate(infos_all):
            if dones_all[i]:
                # 假设你的 env_steps 记录在 info 里或者通过 venv 获取
                # print(f"[探针-终止] 环境 {i} 结束。执行长度: {info.get('exec_len')} | 是否成功: {info.get('is_success')}")
                pass
        
        # 2. 更新当前全局 Batch 字典
        self.current_obs_dicts = obs_batch
        
        combined_rewards = rews_all.copy()
        for i in range(self.num_envs):
            # 处理重采样惩罚 (actions_cache 在 step_async 中保存)
            if self.actions_cache[i] == 0:
                combined_rewards[i] = self.resample_penalty
            
        # 3. 推理下一阶段计划并构建下一帧 RL 观测
        new_plans = self._get_batch_policy_plan(self.current_obs_dicts)
        
        next_rl_obs = np.stack([
            self._make_rl_obs({k: v[i] for k, v in self.current_obs_dicts.items()}, new_plans[i]) 
            for i in range(self.num_envs)
        ])
        
        return next_rl_obs, combined_rewards, dones_all, infos_all

    def reset(self):
        # 1. 获取 Batch 字典 {'key': [N, ...]}
        obs_batch = self.venv.reset()
        self.current_obs_dicts = obs_batch
        
        # 2. 一次性推理出所有环境的计划
        batch_chunks = self._get_batch_policy_plan(self.current_obs_dicts)
        
        # 3. 将 Batch 字典拆分并为每个环境构建 RL 观测
        return np.stack([
            self._make_rl_obs({k: v[i] for k, v in self.current_obs_dicts.items()}, batch_chunks[i]) 
            for i in range(self.num_envs)
        ])

    def _get_batch_policy_plan(self, obs_batch):
        # GPU 批量推理逻辑 (同前，略)
        valid_keys = list(self.algo_instance.obs_shapes.keys())

        # 1. 过滤并转换成 Tensor Batch
        batch_tensor = {}
        for k in valid_keys:
            if k in obs_batch:
                # 将 numpy 数组转为 tensor 并发送到 GPU
                batch_tensor[k] = torch.as_tensor(obs_batch[k]).to(self.device).float()
            
        # 2. 调用 Diffusion Policy 得到动作轨迹 [Batch, Horizon, Action_Dim]
        with torch.no_grad():
            raw_actions = self.algo_instance._get_action_trajectory(batch_tensor)
            
        return raw_actions.detach().cpu().numpy()

    # 其他接口透传...
    def env_is_wrapped(self, wrapper_class, indices=None):
        # 默认返回 False，或者透传给底层的 venv
        return self.venv.env_is_wrapped(wrapper_class, indices=indices)

    def get_attr(self, attr_name, indices=None):
        return self.venv.get_attr(attr_name, indices=indices)

    def set_attr(self, attr_name, value, indices=None):
        return self.venv.set_attr(attr_name, value, indices=indices)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return self.venv.env_method(method_name, *method_args, indices=indices, **method_kwargs)

    def seed(self, seed=None):
        return self.venv.seed(seed)

    def close(self):
        return self.venv.close()



from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

# 1. 定义带 LayerNorm 的网络构建器
class LayerNormMlpExtractor(MlpExtractor):
    def __init__(self, feature_dim, net_arch, activation_fn, device):
        super().__init__(feature_dim, net_arch, activation_fn, device)
        # 重新构建 latent_policy_net (Actor) 和 latent_value_net (Critic)
        # 以包含 LayerNorm
        self.latent_policy = self._build_layer_norm_mlp(feature_dim, net_arch["pi"], activation_fn)
        self.latent_value = self._build_layer_norm_mlp(feature_dim, net_arch["vf"], activation_fn)

    def _build_layer_norm_mlp(self, input_dim, hidden_dims, activation_fn):
        layers = []
        last_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(last_dim, dim))
            layers.append(nn.LayerNorm(dim))  # <--- DSRL 关键 Trick
            layers.append(activation_fn())
            last_dim = dim
        return nn.Sequential(*layers)

# 2. 定义自定义策略类
class DSRLStylePolicy(ActorCriticPolicy):
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = LayerNormMlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )