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

class BatchedAdaptiveWrapper(VecEnv):
    def __init__(self, venv, policy, max_chunk_len, device, reward_offset=0.0, resample_penalty=-0.1, save_all_videos=False, video_dir=None):
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

        self.env_steps = [0] * self.num_envs
        self.max_steps = 600 # 对应你在 train_adaptive.py 里的设置

        # --- 新增视频录制相关变量 ---
        self.save_all_videos = save_all_videos
        self.video_dir = video_dir
        # 为每个环境独立维护帧缓冲区
        self.episode_frames = [[] for _ in range(self.num_envs)]
        self.episode_counts = [0] * self.num_envs
        
        if self.save_all_videos and self.video_dir:
            os.makedirs(self.video_dir, exist_ok=True)

        self.episode_success_flags = [False] * self.num_envs

    def _get_obs_vec(self, obs_dict):
        # 显式定义每个 Key 对应的维度，确保即使缺失也能补齐
        # 这里的维度需要根据你的环境 metadata 确定
        key_shapes = {
            'robot0_eef_pos': (3,),
            'robot0_eef_quat': (4,),
            'robot0_gripper_qpos': (2,),
            'object': (44,) 
        }
        
        vecs = []
        for k, shape in key_shapes.items():
            if k in obs_dict:
                val = obs_dict[k]
                vecs.append(val.flatten().astype(np.float32))
            else:
                # 【核心修复】：如果 Key 缺失，填充对应维度的零向量，而不是跳过
                print(f"[WARNING] Observation key '{k}' missing! Padding with zeros.")
                vecs.append(np.zeros(shape, dtype=np.float32))
                
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
        torch.cuda.empty_cache()
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
            actions_np = raw_actions.detach().cpu().numpy()
            
            # # --- 探针 A ---
            # print(f"[PROBE-A] Model Output Shape: {actions_np.shape}, Expected: (Batch, 16, Dim)")
            return actions_np

    def reset(self):
        obs_batch = self.venv.reset()
        # --- 【修复黑屏】：在第一次 Reset 时强行渲染一帧，唤醒 OpenGL 上下文 ---
        if self.save_all_videos:
            for i in range(self.num_envs):
                _ = self.venv.envs[i].render(mode="rgb_array", height=160, width=160)
        # 打印每个环境抓取的物体坐标（以 'object' 为例，具体 key 视环境而定）
        # print(f"[DEBUG] Reset 初始物体状态 (Env 0): {obs_batch['robot0_eef_pos'][0]}")
        # eef_pos = raw_obs.get('robot0_eef_pos')
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

            # # --- 探针 B ---
            current_buffer = self.current_chunks[i]
            buffer_len = len(current_buffer) if current_buffer is not None else -1
            # print(f"[PROBE-B] Env {i} | Requested k: {k} | Buffer Type: {type(current_buffer)} | Buffer Len: {buffer_len}")
            
            if k > 0 and (current_buffer is None or buffer_len < k):
                print(f"!!! [CRITICAL] Env {i} 出现异常：RL 请求执行 {k} 步，但缓冲区只有 {buffer_len} 步！")
                # 这里可以加个断点或强制打印出 buffer 的内容
            
            # 场景 A：RL 选了重采样 (k=0)
            if k == 0:
                rewards[i] = self.resample_penalty
                infos[i] = {"exec_len": 0, "resampled": True}
                continue
            
            # 场景 B：执行 k 步物理步
            steps_actually_run = 0
            # 【探针变量】：专门用来锁定包含统计信息的 info
            # captured_episode_info = None
            last_sub_info = {}
            # last_info = {} # 用于保存最后一步的 info
            # 注意：此时 current_chunks[i] 是上一轮决策后生成的全新 16 步
            for step_idx in range(k):
                # --- 探针 C ---
                if len(self.current_chunks[i]) == 0:
                    print(f"!!! [ERROR] 崩溃点：Env {i} 物理循环第 {step_idx}/{k} 步时 Chunk 空了")
                    # 重点查看：此时的 step_idx 是多少？如果 step_idx < k，说明确实没给够
                    break
                # # 理论上 current_chunks 不会在此为空，但做个稳健性检查
                # if len(self.current_chunks[i]) == 0:
                #     print("!!! [ERROR] 当前动作块已空，无法继续执行！")
                #     break
                
                act = self.current_chunks[i][0]
                self.current_chunks[i] = self.current_chunks[i][1:] # 消耗一个动作

                
                next_obs, r, d, truncated, sub_info = self.venv.envs[i].step(act)

                # --- 【核心修改】：在物理步内捕获每一帧，确保视频丝滑 ---
                if self.save_all_videos:
                    # 获取当前物理帧 (需要 render_offscreen=True)
                    frame = self.venv.envs[i].render(mode="rgb_array", height=160, width=160)
                    self.episode_frames[i].append(frame)


                self.env_steps[i] += 1 # 物理步计数
                last_sub_info = sub_info

                # # --- 探针 1：实时监控 Monitor 信号 ---
                # if 'episode' in sub_info:
                #     print(f"!!! [探针-发现信号] Env {i} 在物理步 {self.env_steps[i]} 产生 episode 数据: {sub_info['episode']}")
                #     captured_episode_info = sub_info.copy() # 立即锁定，防止被下一帧覆盖

                # # 每一百步打印一次心跳，确认环境还在跑，并观察步数是否超标
                # if self.env_steps[i] % 20 == 0:
                #     print(f"[HEARTBEAT] Env {i} Step: {self.env_steps[i]}, Done: {d}, Reward: {r}")


                # if d or truncated:
                #     dones[i] = True
                #     self.env_steps[i] = 0 # 重置计数


                # if d:
                #     print(f"!!! [SUCCESS/DONE] 底层环境 i={i} 触发了 done! Reward: {r}, Info keys: {sub_info.keys()}")
                #     if 'episode' in sub_info:
                #         print(f">>> Monitor 数据已生成: {sub_info['episode']}")
                # # ---------------------

                # if truncated:
                #     print(f"!!! [TRUNCATED] 底层环境 i={i} 触发了 truncated! Reward: {r}, Info keys: {sub_info.keys()}")
                #     if 'episode' in sub_info:
                #         print(f">>> Monitor 数据已生成: {sub_info['episode']}")

                #### debug#### debug#### debug



                rewards[i] += (r - self.reward_offset)
                self.current_obs_dicts[i] = next_obs # 更新该环境的最新观测
                steps_actually_run += 1

                # 【核心改进】：检测成功并提前终止
                # 检查 reward 是否达到 1.0 或者 info 里是否有 success 标志
                is_success = False
                if r >= 1.0:
                    is_success = True
                elif isinstance(sub_info.get('is_success'), dict):
                    is_success = sub_info['is_success'].get('task', False)
                elif sub_info.get('success'):
                    is_success = True

                if is_success:
                #     print(f"!!! [SUCCESS EARLY EXIT] Env {i} 在第 {self.env_steps[i]} 步成功，提前终止回合")
                    self.episode_success_flags[i] = True # 标记该回合已成功
                    d = True # 强行设为 done


                # 3. 检查是否达到最大步数 (600步)
                if self.env_steps[i] >= self.max_steps:
                    # print(f"!!! [FORCE DONE] Env {i} 达到手动上限 {self.max_steps}")
                    d = True
                    truncated = True

                # 回合结束处理
                if d or truncated:
                    dones[i] = True
                    # --- 【核心修改】：轨迹结束，保存该环境的视频 ---
                    if self.save_all_videos and len(self.episode_frames[i]) > 0:
                        self._save_video(i, success=self.episode_success_flags[i])
                    # 关键：保存完视频后，重置该环境的成功标志
                    self.episode_success_flags[i] = False
                    break



                # last_info = sub_info # 持续更新，保留最后一步包含 Monitor 数据的 info
                
                # if d or truncated:
                #     dones[i] = True
                #     break

            # # --- 探针 2：验证数据合并 ---
            # if captured_episode_info:
            #     # 确保把抓到的核心数据更新进去
            #     infos[i].update(captured_episode_info)
            # else:
            #     if dones[i]:
            #         print(f"??? [探针-异常] Env {i} 回合已结束，但全程未抓到 'episode' 键！sub_info 所有键: {sub_info.keys()}")

            infos[i].update(last_sub_info)
            # 清洗 is_success (防止上一轮报错)
            if 'is_success' in infos[i] and isinstance(infos[i]['is_success'], dict):
                infos[i]['is_success'] = infos[i]['is_success'].get('task', False)
            
            # 【关键修复】：合并底层 info，确保 Monitor 的 episode 数据能传给 PPO
            # infos[i].update(last_info)


            infos[i].update({"exec_len": steps_actually_run, "resampled": False})

            

        # 2. 核心逻辑：强制抛弃剩余计划，全员刷新 (MPC 模式)
        needs_new = []
        for i in range(self.num_envs):
            if dones[i]:
                # 重置计数器
                self.env_steps[i] = 0
                # 环境结束，手动重置环境获取初始状态
                raw_obs, _ = self.venv.envs[i].reset()
                self.current_obs_dicts[i] = raw_obs

                # --- 在这里加上打印 ---
                # obj_pos = raw_obs.get('object', np.array([0,0,0]))[:3]
                # eef_pos = raw_obs.get('robot0_eef_pos')
                # print(f"[DEBUG] 轨迹结束，自动重置 Env {i}。新物体位置: {eef_pos}")
            
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

        # # --- 探针 3：最终输出前检查 ---
        # if any(dones):
        #     for i, info in enumerate(infos):
        #         if dones[i]:
        #             has_ep = 'episode' in info
        #             print(f"[最终检查] Env {i} 准备提交给 PPO: Done={dones[i]}, 包含 episode={has_ep}")
        
        return next_rl_obs, rewards, dones, infos

    def _save_video(self, env_idx, success=False):
        """内部辅助函数：保存视频并清空缓存"""
        self.episode_counts[env_idx] += 1
        # 根据成功与否添加后缀
        status_str = "success" if success else "fail"
        filename = f"env_{env_idx}_ep_{self.episode_counts[env_idx]:03d}_{status_str}.mp4"
        path = os.path.join(self.video_dir, filename)
        
        try:
            # fps=20 对应 robosuite 的标准速度，看起来比较自然
            imageio.mimsave(path, self.episode_frames[env_idx], fps=20)
        except Exception as e:
            print(f"[ERROR] 保存视频失败: {e}")
            
        # 必须清空，否则内存会爆
        self.episode_frames[env_idx] = []

    # 为了 SB3 兼容性，确保 render 请求能透传
    def render(self, mode="rgb_array", height=512, width=512):
        return self.venv.render(mode=mode, height=height, width=width)

    def close(self): self.venv.close()
    def get_attr(self, a, i=None): return self.venv.get_attr(a, i)
    def set_attr(self, a, v, i=None): return self.venv.set_attr(a, v, i)
    def env_method(self, m, *as_, indices=None, **ks): return self.venv.env_method(m, *as_, indices=indices, **ks)
    def env_is_wrapped(self, w, i=None): return [False] * self.num_envs
