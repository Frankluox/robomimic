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

    def render(self, mode="human"): return self.env.render(mode=mode)

class BatchedAdaptiveWrapper(VecEnv):
    def __init__(self, venv, policy, max_chunk_len, device, reward_offset=0.0, resample_penalty=-0.1):
        self.venv = venv
        self.num_envs = venv.num_envs
        self.policy = policy
        self.max_chunk_len = max_chunk_len
        self.device = device
        self.reward_offset = reward_offset
        self.resample_penalty = resample_penalty
        
        self.algo_instance = policy.policy if hasattr(policy, 'policy') else policy
        self.action_dim = self.algo_instance.ac_dim
        
        rl_action_space = spaces.Discrete(max_chunk_len + 1)
        tmp_obs_batch = venv.reset()
        # 处理 DummyVecEnv 返回的 Batch 字典
        sample_obs_dict = {k: v[0] for k, v in tmp_obs_batch.items()}
        # 1. 计算状态向量维度 (EEF + Gripper + Object)
        obs_vec = self._get_obs_vec(sample_obs_dict)
        self.obs_dim = obs_vec.shape[0] 
        
        # 2. 计算总维度 (状态向量 + 动作块向量)
        # 动作块向量 = 16 步 * 动作维度 (如 7)
        self.total_obs_dim = self.obs_dim + (max_chunk_len * self.action_dim)
        
        # 定义完整的 RL 观测空间
        rl_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.total_obs_dim,), dtype=np.float32
        )
        super().__init__(self.num_envs, rl_observation_space, rl_action_space)

        self.current_obs_dicts = [None] * self.num_envs
        self.current_chunks = [None] * self.num_envs
        self.actions_cache = None

        # 在 BatchedAdaptiveWrapper 的 __init__ 中加入
        print("-" * 30)
        print("[DEBUG] Diffusion Policy 期望的输入键值 (Obs Keys):")
        # self.algo_instance 是之前代码里拿到的 policy.policy
        for key in self.algo_instance.obs_shapes.keys():
            print(f"Key: {key}, Shape: {self.algo_instance.obs_shapes[key]}")
        print("-" * 30)

    def _get_obs_vec(self, obs_dict):
        """
        同步 DP 模型视野：将 44 维的 object 信息加入 RL 观测
        """
        # 定义核心物理状态键值
        target_keys = [
            'robot0_eef_pos',      # [3]
            'robot0_eef_quat',     # [4]
            'robot0_gripper_qpos', # [2]
            'object'               # [44] <- 核心情报补完
        ]
        
        vecs = []
        for k in target_keys:
            if k in obs_dict:
                val = obs_dict[k]
                # 展平并确保是 float32
                vecs.append(val.flatten().astype(np.float32))
            else:
                # 容错：如果某个 Key 没拿到，打印警告（仅限初期调试）
                pass 
                
        return np.concatenate(vecs)

    def _make_rl_obs(self, obs_dict, chunk):
        obs_vec = self._get_obs_vec(obs_dict)
        if chunk.shape[0] < self.max_chunk_len:
            pad = np.zeros((self.max_chunk_len - chunk.shape[0], self.action_dim))
            chunk = np.concatenate([chunk, pad], axis=0)
        return np.concatenate([obs_vec, chunk.flatten()]).astype(np.float32)

    def _get_batch_policy_plan(self, obs_data):
        """
        核心修复：使用源码中存在的 list_of_flat_dict_to_dict_of_list 手动实现 Batching
        """
        valid_keys = list(self.algo_instance.obs_shapes.keys())
        
        if isinstance(obs_data, list):
            # 1. 过滤 Key：只保留模型需要的输入
            filtered = [{k: d[k] for k in valid_keys if k in d} for d in obs_data]
            
            # 2. 使用你源码中有的函数：将 [dict, dict] 转换为 {key: [val, val]}
            dict_of_lists = TensorUtils.list_of_flat_dict_to_dict_of_list(filtered)
            
            # 3. 手动将 List of Arrays 转换为 Stacked Tensor (Batch)
            batch_obs = {}
            for k, val_list in dict_of_lists.items():
                # 将列表中的 numpy 数组转换为 torch tensors 并堆叠
                tensors = [torch.as_tensor(item) for item in val_list]
                batch_obs[k] = torch.stack(tensors)
        else:
            # reset() 的情况，已经是 Dict[str, np.ndarray]，直接转换即可
            filtered = {k: obs_data[k] for k in valid_keys if k in obs_data}
            batch_obs = TensorUtils.to_tensor(filtered)
            
        # 4. 发送到设备并转为浮点数
        batch_obs = TensorUtils.to_device(batch_obs, self.device)
        batch_obs = TensorUtils.to_float(batch_obs)
        
        with torch.no_grad():
            # 调用 Diffusion Policy 得到动作轨迹 [Batch, Horizon, Action_Dim]
            raw_actions = self.algo_instance._get_action_trajectory(batch_obs)
            return raw_actions.detach().cpu().numpy()

    def reset(self):
        obs_batch = self.venv.reset()
        batch_chunks = self._get_batch_policy_plan(obs_batch)
        self.current_obs_dicts = [{k: v[i] for k, v in obs_batch.items()} for i in range(self.num_envs)]
        self.current_chunks = [batch_chunks[i] for i in range(self.num_envs)]
        return np.stack([self._make_rl_obs(self.current_obs_dicts[i], self.current_chunks[i]) for i in range(self.num_envs)])

    def step_async(self, actions): self.actions_cache = actions

    def step_wait(self):
        actions = self.actions_cache
        # 初始化返回数据
        rewards = np.zeros(self.num_envs)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = [{} for _ in range(self.num_envs)]
        
        # 1. 执行周期：物理环境跑 k 步
        for i in range(self.num_envs):
            k = int(actions[i])
            
            # 场景 A：RL 选了重采样 (k=0)
            if k == 0:
                rewards[i] = self.resample_penalty
                infos[i] = {"exec_len": 0, "resampled": True}
                continue
            
            # 场景 B：执行 k 步物理步
            steps_actually_run = 0
            # 注意：此时 current_chunks[i] 是上一轮决策后生成的全新 16 步
            for _ in range(k):
                # 理论上 current_chunks 不会在此为空，但做个稳健性检查
                if len(self.current_chunks[i]) == 0:
                    break
                
                act = self.current_chunks[i][0]
                self.current_chunks[i] = self.current_chunks[i][1:] # 消耗一个动作
                
                next_obs, r, d, _, _ = self.venv.envs[i].step(act)
                rewards[i] += (r - self.reward_offset)
                self.current_obs_dicts[i] = next_obs # 更新该环境的最新观测
                steps_actually_run += 1
                
                if d:
                    dones[i] = True
                    break
            
            infos[i] = {"exec_len": steps_actually_run, "resampled": False}

        # 2. 核心逻辑：强制抛弃剩余计划，全员刷新 (MPC 模式)
        needs_new = []
        for i in range(self.num_envs):
            if dones[i]:
                # 环境结束，手动重置环境获取初始状态
                raw_obs, _ = self.venv.envs[i].reset()
                self.current_obs_dicts[i] = raw_obs
            
            # 重点：无论刚才跑了多少步，为了实现“每步重决策”，
            # 我们将所有环境加入 needs_new 列表进行全量重采样
            needs_new.append(i)

        # 调用 GPU 进行批量 Diffusion 推理
        if needs_new:
            obs_to_infer = [self.current_obs_dicts[i] for i in needs_new]
            new_plans = self._get_batch_policy_plan(obs_to_infer)
            for idx, env_idx in enumerate(needs_new):
                # 无论之前剩了多少步，全部覆盖为全新的 16 步
                self.current_chunks[env_idx] = new_plans[idx]

        # 3. 构造返回给 RL 的下一帧观测
        # 此时返回的 chunk 必然是刚生成的、完整的 16 步
        next_rl_obs = np.stack([
            self._make_rl_obs(self.current_obs_dicts[i], self.current_chunks[i]) 
            for i in range(self.num_envs)
        ])
        
        return next_rl_obs, rewards, dones, infos

    def close(self): self.venv.close()
    def get_attr(self, a, i=None): return self.venv.get_attr(a, i)
    def set_attr(self, a, v, i=None): return self.venv.set_attr(a, v, i)
    def env_method(self, m, *as_, indices=None, **ks): return self.venv.env_method(m, *as_, indices=indices, **ks)
    def env_is_wrapped(self, w, i=None): return [False] * self.num_envs

class AdaptiveChunkingWrapper(gym.Env):
    def __init__(self, env, policy, max_chunk_len, device, reward_offset=0.0, resample_penalty=-0.1):
        """
        Args:
            env: 原始 Robomimic 环境 (step返回 obs_dict)
            policy: 预训练好的 Diffusion Policy (frozen)
            max_chunk_len: N, Diffusion Policy 输出的最大长度
            resample_penalty: 当 RL 选择 k=0 (重采样) 时的惩罚值
        """
        self.env = env
        self.policy = policy
        self.max_chunk_len = max_chunk_len
        self.device = device
        self.reward_offset = reward_offset
        self.resample_penalty = resample_penalty

        # 定义需要拼接的 Observation Keys
        self.target_obs_keys = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']
        
        # 1. Action Space: 0 到 N (共 N+1 个离散动作)
        # 0: 重采样, 1-N: 执行对应步数
        self.action_space = spaces.Discrete(max_chunk_len + 1)
        
        # 2. Observation Space: [Flat_Obs, Flat_Action_Chunk]
        # 先获取一次 obs 看看维度
        raw_obs = self.env.reset()
        # print(f"[AdaptiveChunkingWrapper] 原始 Obs Keys: {list(tmp_obs.keys())}")
        # 兼容处理 reset 返回 (obs, info) 的情况
        if isinstance(raw_obs, tuple):
            raw_obs = raw_obs[0]
        obs_vec = self._get_obs_vec(raw_obs)
        self.obs_dim = obs_vec.shape[0]

        #
        # 注意：这里 policy 可能是 RolloutPolicy，也可能是算法实例
        # 我们的 train_adaptive.py 已经处理了 RolloutPolicy，这里直接用 ac_dim
        if hasattr(policy, 'ac_dim'):
            self.action_dim = policy.ac_dim
        else:
            self.action_dim = policy.policy.ac_dim

        self.chunk_flat_dim = max_chunk_len * self.action_dim
        
        self.total_obs_dim = self.obs_dim + self.chunk_flat_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.total_obs_dim,), dtype=np.float32
        )
        
        self.current_obs_dict = raw_obs
        self.current_chunk = None # shape (T, Da)

    def _get_obs_vec(self, obs_dict):
        """
        辅助函数：将特定的 observation keys 拼接成一个扁平的向量。
        """
        # 按照 target_obs_keys 的顺序拼接向量
        vecs = []
        for k in self.target_obs_keys:
            if k in obs_dict:
                val = obs_dict[k]
                # 确保是 1D 数组
                if isinstance(val, np.ndarray):
                    vecs.append(val.flatten())
                else:
                    # 处理 scalar 的情况
                    vecs.append(np.array([val]))
            else:
                raise KeyError(f"Observation key '{k}' not found in environment output! Available keys: {list(obs_dict.keys())}")
        
        return np.concatenate(vecs).astype(np.float32)

    def _get_policy_plan(self, obs_dict):
        """调用 Frozen Policy 生成 Action Chunk"""
        # 预处理 obs
        obs_dict = TensorUtils.to_tensor(obs_dict)
        obs_dict = TensorUtils.to_batch(obs_dict) # [B, ...]
        obs_dict = TensorUtils.to_device(obs_dict, self.device)
        obs_dict = TensorUtils.to_float(obs_dict)
        
        with torch.no_grad():
            # 使用 _get_action_trajectory 获取完整序列
            # 注意：需要确保 policy 处于 eval 模式，但如果为了重采样需要随机性，
            # Diffusion Policy 本身的去噪过程是随机的，所以每次调用都会不同。

            # 如果是 RolloutPolicy，调用其内部的 .policy 属性
            if hasattr(self.policy, 'policy'):
                raw_action = self.policy.policy._get_action_trajectory(obs_dict)
            else:
                raw_action = self.policy._get_action_trajectory(obs_dict)
            # raw_action = self.policy._get_action_trajectory(obs_dict)
            raw_action = raw_action.detach().cpu().numpy()[0] # [T, Da]
            
        return raw_action

    def _make_rl_obs(self, obs_dict, action_chunk):
        """拼接拼接后的 Obs 向量和 Chunk 作为 RL 的输入"""
        obs_vec = self._get_obs_vec(obs_dict)

        # 如果 chunk 长度不足 N (虽然 diffusion 一般是固定的)，padding 0
        current_len = action_chunk.shape[0]
        if current_len < self.max_chunk_len:
            padding = np.zeros((self.max_chunk_len - current_len, self.action_dim))
            action_chunk = np.concatenate([action_chunk, padding], axis=0)
        
        flat_chunk = action_chunk.flatten()
        return np.concatenate([obs_vec, flat_chunk]).astype(np.float32)

    def reset(self, **kwargs):
        raw_obs = self.env.reset()
        if isinstance(raw_obs, tuple):
            raw_obs = raw_obs[0]

        self.current_obs_dict = raw_obs
        
        # 初始推理
        self.current_chunk = self._get_policy_plan(self.current_obs_dict)

        rl_obs = self._make_rl_obs(self.current_obs_dict, self.current_chunk)
        
        # === 核心修正：返回 (观测, 信息字典) ===
        return rl_obs, {}

    def step(self, action):
        k = int(action) # RL 输出的长度
        
        # === Case 0: Resample (重采样) ===
        if k == 0:
            # 给予惩罚，强制策略去寻找更好的 chunk 或者学会执行
            reward = self.resample_penalty
            terminated = False # 对应原本的 done
            truncated = False  # 新增的截断标志
            info = {"exec_len": 0, "resampled": True}
            
            # 环境状态不变，但重新请求 Policy 生成新的 Chunk
            # Diffusion Policy 的生成过程包含随机噪声，所以结果会变
            self.current_chunk = self._get_policy_plan(self.current_obs_dict)
            
            # 返回：旧的 Obs + 新的 Chunk
            rl_obs = self._make_rl_obs(self.current_obs_dict, self.current_chunk)
            # 返回 5 个值
            return rl_obs, reward, terminated, truncated, info

        # === Case > 0: Execute k steps (执行 k 步) ===
        total_reward = 0
        terminated = False
        truncated = False
        info = {"exec_len": k, "resampled": False}
        
        # 截取前 k 步
        # 注意边界检查，虽然理论上 RL 不会输出 > N，但防止万一
        steps_to_run = min(k, len(self.current_chunk))
        
        for i in range(steps_to_run):
            act = self.current_chunk[i]
            next_obs, r, d, _ = self.env.step(act)
            
            # 累加奖励 (包含 reward_offset)
            total_reward += (r - self.reward_offset)
            self.current_obs_dict = next_obs
            
            if d:
                done = True
                break
        
        # 只有当环境没结束时，才进行下一次推理
        if not terminated:
            self.current_chunk = self._get_policy_plan(self.current_obs_dict)
            rl_obs = self._make_rl_obs(self.current_obs_dict, self.current_chunk)
        else:
            # 环境结束，给个全0的chunk占位，外部会调用reset
            dummy_chunk = np.zeros((self.max_chunk_len, self.action_dim))
            rl_obs = self._make_rl_obs(self.current_obs_dict, dummy_chunk)

        return rl_obs, total_reward, terminated, truncated, info