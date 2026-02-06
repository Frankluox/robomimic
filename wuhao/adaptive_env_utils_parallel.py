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
import collections # 顶部增加导入

import pickle # 新增
import copy   # 新增

import threading # 顶部增加

def _async_save_worker(vid_path, frames, data_path, log_data, save_videos, save_data):
    """ 在后台线程中执行沉重的写磁盘操作 """
    try:
        import imageio
        import pickle
        if save_videos and frames:
            imageio.mimsave(vid_path, frames, fps=20)
        if save_data and log_data:
            with open(data_path, 'wb') as f:
                pickle.dump(log_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"[ERROR] Async save failed: {e}")


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
# Modified: Internal wrapper supporting Gamma Discounting
# =============================================================================
class InternalVariableChunkWrapper(gym.Wrapper):
    def __init__(self, env, reward_offset=0.0, max_steps=600, save_videos=False, save_data=False, video_dir=None, env_id=0, gamma=0.99):
        super().__init__(env)
        self.reward_offset = reward_offset
        self.max_steps = max_steps
        self.save_videos = save_videos
        self.save_data = save_data # <--- [修改] 保存开关
        self.video_dir = video_dir
        self.env_id = env_id  # 用于区分不同进程的文件名
        self.gamma = gamma  # <--- [SMDP] Receive Gamma

        self.current_step = 0
        self.episode_count = 0
        self.episode_frames = []
        self.current_obs = None
        self.success_flag = False

        # [修改] 仅当需要保存数据时才初始化日志
        self.episode_log = None
        if self.save_data:
            self.episode_log = {
                'trajectory': [], 
                'decisions': [] 
            }

        if (self.save_videos or self.save_data) and self.video_dir:
            os.makedirs(self.video_dir, exist_ok=True)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_step = 0 # 重置计数器
        self.episode_frames = []
        self.success_flag = False
        self.current_obs = obs

        # [修改] 重置日志
        if self.save_data:
            self.episode_log = {
                'trajectory': [], 
                'decisions': []
            }

        return obs, info

    def step(self, action_data):
        try:
            chunk = action_data['chunk']
            k = int(action_data['k'])
            dist_probs = action_data.get('dist_probs', None)

            # [修改] 仅在开启 save_data 时记录决策
            if self.save_data:
                self.episode_log['decisions'].append({
                    'step_start': self.current_step,
                    'chosen_k': k,
                    'policy_dist': dist_probs 
                })

            # 探针：记录执行前的底层物理步数 (针对 Robosuite)
            base_env = self.env.env
            step_before = base_env.cur_time if hasattr(base_env, 'cur_time') else 0
            
            if k <= 0:
                # 修改点：确保 resampled 时也返回 info 字典，且 success_flag 状态保持
                info = {"exec_len": 0, "resampled": True, "is_success": self.success_flag}
                # [探针-重采样]
                # if self.episode_count == 0 and self.current_step < 500: # 只打印第一个 Episode 的前 50 步
                    # print(f"\n[PROBE-ENV] Step {self.current_step}: Resample triggered (k=0). Reward Penalty: 0.0 (Applied later in VecEnv)")
                # k=0 means 0 time passed, so discount is 1.0. Reward is just penalty.
                return self.current_obs, 0.0, False, False, info

            total_discounted_reward = 0.0
            current_step_gamma = 1.0 # gamma^0

            # 定义两个标志位
            is_terminated = False
            is_truncated = False
            actual_steps = 0

            

            # [探针-Chunk开始]
            probe_log = []
            
            for i in range(k):

                # 1. 统一获取当前帧 (如果需要保存任何数据)
                curr_frame = None

                if self.save_data or self.save_videos:
                    # 显式使用 .copy() 确保数据独立，避免缓冲区被底层环境修改
                    raw_frame = self.env.render(mode="rgb_array", height=160, width=160)
                    curr_frame = np.array(raw_frame).copy()

                # 2. 记录到数据日志
                if self.save_data:
                    step_record = {
                        'step': self.current_step,
                        'obs': copy.deepcopy(self.current_obs),
                        'action': chunk[i].copy(),
                        'image': curr_frame # 直接复用已渲染的帧
                    }
                    self.episode_log['trajectory'].append(step_record)


                obs, reward, term, trunc, info = self.env.step(chunk[i])
                self.current_step += 1 # 物理步数累加
                actual_steps += 1

                # --- [SMDP] Calculate Discounted Reward Sum ---
                # R_chunk = r_0 + gamma * r_1 + gamma^2 * r_2 ...
                r_t = (reward - self.reward_offset)
                discounted_r_t = r_t * current_step_gamma
                total_discounted_reward += discounted_r_t

                # # [探针-物理步细节] 记录每一步的原始奖励、当前衰减因子、衰减后奖励
                # if self.episode_count == 0 and self.current_step < 50:
                #     probe_log.append(f"  t={i}: r_raw={r_t:.4f} * gam={current_step_gamma:.4f} -> {discounted_r_t:.4f}")

                # Update gamma for next step
                current_step_gamma *= self.gamma
                # -----------------------------------------------



                self.current_obs = obs

                # --- 录制逻辑 ---
                if self.save_videos and curr_frame is not None:
                    self.episode_frames.append(curr_frame)

                # 判定成功（Robomimic 典型判定）
                if reward >= 1.0 or (isinstance(info.get('is_success'), dict) and info['is_success'].get('task')):
                    self.success_flag = True

                # --- 核心逻辑修正 ---
                # 1. 底层环境触发
                if term: is_terminated = True
                if trunc: is_truncated = True

                # 2. 成功触发 (视为 terminated)
                if self.success_flag:
                    is_terminated = True

                # 3. 步数超限触发 (视为 truncated)
                if self.current_step >= self.max_steps:
                    # 只有在还没成功、也没死的情况下，才算超时
                    if not is_terminated:
                        is_truncated = True
                        info["TimeLimit.truncated"] = True # 显式注入

                # 只要任意一个为 True，由于是大循环，都要 break
                if is_terminated or is_truncated:
                    break



            # 探针：记录执行后的物理步数
            step_after = base_env.cur_time if hasattr(base_env, 'cur_time') else 0
            actual_physics_steps = actual_steps # 循环里的计数
            
            # 验证：物理步数增长必须等于实际循环次数
            # 如果物理步数增长大于 k，说明有隐形步进
            # print(f"[探针-物理时钟] 环境ID: {os.getpid()} | 请求k: {k} | 实际步进: {actual_physics_steps} | 物理时间增量: {step_after - step_before}")
            # if self.success_flag:
                # print(f"[探针-物理时钟] 环境ID: {os.getpid()} | 请求k: {k} | 实际步进: {actual_physics_steps} | 物理时间增量: {step_after - step_before}")
            

            # [探针-Chunk结算] 打印汇总
            # if len(probe_log) > 0:
            #     print(f"\n[PROBE-ENV] Step {self.current_step - actual_steps} -> {self.current_step} (k={actual_steps}):")
            #     print("\n".join(probe_log))
            #     print(f"  => Chunk Reward SMDP Sum: {total_discounted_reward:.4f}")

            # --- 核心修正：显式写回成功信号到 info ---
            # 这样主进程的 SuccessBestModelCallback 才能看到它
            info["is_success"] = self.success_flag

            info["exec_len"] = actual_steps
            info["resampled"] = False

            if is_terminated or is_truncated:
                # [修改] 根据开关分别保存
                self._handle_episode_done()
                self.episode_count += 1
                # 探针：记录 done 瞬间的末尾坐标
                pos_at_done = self.current_obs.get('robot0_eef_pos', np.zeros(3)).copy()
                
                # 子进程会自动触发底层重置 (如果是 VecEnv 包装的话)
                # 或者如果是手动重置，在此观察
                # print(f"[探针-重置] 环境 {os.getpid()} 触发 Done. 终止坐标: {pos_at_done}")
            
            return self.current_obs, total_discounted_reward, is_terminated, is_truncated, info
        except Exception as e:
            # 【关键】：在子进程死掉前，强制把错误打印出来
            import traceback
            print(f"\n[FATAL ERROR] Worker {self.env_id} crashed!")
            traceback.print_exc()
            # 即使崩溃也尝试释放内存，防止拖死整个系统
            self.episode_frames = [] 
            raise e # 重新抛出让主进程感知

    def _handle_episode_done(self):
        try:
            status = "success" if self.success_flag else "fail"
            
            # 准备文件名
            vid_path = os.path.join(self.video_dir, f"env_{self.env_id}_ep_{self.episode_count:03d}_{status}.mp4")
            data_path = os.path.join(self.video_dir, f"env_{self.env_id}_ep_{self.episode_count:03d}_{status}_data.pkl")
            
            # 立即复制数据并清空原列表，确保主进程可以继续
            frames_to_save = self.episode_frames
            self.episode_frames = []
            log_to_save = self.episode_log
            if self.save_data:
                self.episode_log = {'trajectory': [], 'decisions': []}

            # 启动后台线程保存，不阻塞当前 worker 的重置和下一轮渲染
            t = threading.Thread(
                target=_async_save_worker, 
                args=(vid_path, frames_to_save, data_path, log_to_save, self.save_videos, self.save_data)
            )
            t.start()

        except Exception as e:
            print(f"[ERROR] Worker {self.env_id} failed to trigger async save: {e}")
            


    # def _save_video(self):
    #     import imageio
    #     status = "success" if self.success_flag else "fail"
    #     filename = f"env_{self.env_id}_ep_{self.episode_count:03d}_{status}.mp4"
    #     path = os.path.join(self.video_dir, filename)
    #     try:
    #         imageio.mimsave(path, self.episode_frames, fps=20)
    #     except Exception as e:
    #         print(f"[ERROR] 子进程 {self.env_id} 保存视频失败: {e}")
    #     finally:
    #         # 【必须】：无论成功失败，必须清空，否则内存会爆炸导致 Broken Pipe
    #         self.episode_frames = []


# =============================================================================
# 修改后的主进程包装器
# =============================================================================
class BatchedAdaptiveWrapper(VecEnv):
    def __init__(self, venv, policy, max_chunk_len, device, reward_offset=0.0, resample_penalty=-0.1, save_all_videos=False, video_dir=None, 
                 action_history_len=0):
        self.venv = venv
        self.num_envs = venv.num_envs
        self.policy = policy
        self.max_chunk_len = max_chunk_len
        self.device = device
        self.resample_penalty = resample_penalty
        self.action_history_len = action_history_len # <--- 存储窗口长度
        
        self.algo_instance = policy.policy if hasattr(policy, 'policy') else policy
        self.action_dim = self.algo_instance.ac_dim
        
        rl_action_space = spaces.Discrete(max_chunk_len + 1)
        
        # 统计数据加载（用于归一化）
        self.stats = np.load("/home/wuhao/jobspace/luoxu/robomimic/datasets/toolhang/toolhang.npz")
        
        # 探测观测空间维度
        tmp_obs_batch = venv.reset()
        sample_obs_dict = tmp_obs_batch[0] if isinstance(tmp_obs_batch, list) else {k:v[0] for k,v in tmp_obs_batch.items()}
        obs_vec = self._get_obs_vec(sample_obs_dict)

        # --- 【修改】：重新计算 RL 观测空间维度 ---
        # 维度 = 物理观测 + 动作历史 (len * dim) + 当前预测 Chunk (max_len * dim)
        self.total_obs_dim = obs_vec.shape[0] + (self.action_history_len * self.action_dim) + (max_chunk_len * self.action_dim)
        # self.total_obs_dim = obs_vec.shape[0] + (max_chunk_len * self.action_dim)
        
        rl_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.total_obs_dim,), dtype=np.float32)
        super().__init__(self.num_envs, rl_observation_space, rl_action_space)

        self.current_obs_dicts = [None] * self.num_envs
        self.actions_cache = None
        self.last_chunks = [None] * self.num_envs # 记录上一轮生成的计划，用于更新历史

        # --- 【新增】：初始化动作历史缓冲区 ---
        # 使用 numpy 数组存储每个环境的动作历史 [num_envs, history_len, action_dim]
        self.action_histories = np.zeros((self.num_envs, self.action_history_len, self.action_dim), dtype=np.float32)

        # [新增] 暂存 Policy 分布
        self.next_step_dists = None

    # [新增] 外部调用接口
    def set_next_step_dists(self, dists):
        """dists: numpy array (num_envs, num_actions)"""
        self.next_step_dists = dists

    def _update_action_history(self, env_idx, executed_actions):
        """
        将实际执行的动作推入历史窗口（类似于 Queue 的滑动窗口）
        """
        if self.action_history_len <= 0:
            return
            
        num_new = len(executed_actions)
        if num_new == 0:
            return
            
        if num_new >= self.action_history_len:
            # 如果新动作比窗口还长，直接取最后段
            self.action_histories[env_idx] = executed_actions[-self.action_history_len:]
        else:
            # 经典的滑动窗口更新：旧数据左移，新数据补右侧
            self.action_histories[env_idx] = np.roll(self.action_histories[env_idx], -num_new, axis=0)
            self.action_histories[env_idx][-num_new:] = executed_actions

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

        # 建议在这里也加上，防止异常值传递给后续逻辑
        final_vec = np.clip(final_vec, -10.0, 10.0)
        
        # 探针：检查归一化后的量级
        if np.max(np.abs(final_vec)) > 20.0:
            key_name = "unknown" # 可以进一步定位是哪个 key 没归一化好
            # print(f"[探针-异常警告] 归一化后的特征量级过大 (>20)! 检查统计量对齐。Max: {np.max(final_vec):.2f}")
            
        return final_vec

    def _make_rl_obs(self, env_idx, obs_dict, chunk):
        obs_vec = self._get_obs_vec(obs_dict)
        action_mean = self.stats["action_mean"]
        action_std = self.stats["action_std"]

        # 1. 处理动作历史 (归一化)
        history_part = np.array([])
        if self.action_history_len > 0:
            norm_history = (self.action_histories[env_idx] - action_mean) / action_std
            history_part = norm_history.flatten()

        # 2. 处理当前预测 Chunk (归一化)
        normalized_chunk = (chunk - action_mean) / action_std
        if normalized_chunk.shape[0] < self.max_chunk_len:
            pad = np.zeros((self.max_chunk_len - normalized_chunk.shape[0], self.action_dim))
            normalized_chunk = np.concatenate([normalized_chunk, pad], axis=0)
        chunk_part = normalized_chunk.flatten()

        # 3. 拼接：[Obs, History, Chunk]
        final_vec = np.concatenate([obs_vec, history_part, chunk_part]).astype(np.float32)
        

        if np.any(np.abs(final_vec) > 50): 
            print(f"[WARN] Obs normalization outlier: max value {np.max(np.abs(final_vec))}")
        return final_vec

    def step_async(self, actions):
        requested_ks = actions.astype(int)
        # 主进程在 GPU 上批量计算 Chunk
        # batch_chunks = self._get_batch_policy_plan(self.current_obs_dicts)
        
        # 探针：观察 Batch 中的 k 分布
        # print(f"[探针-决策] 本轮 Batch 长度分配: {requested_ks.tolist()} | Max k: {np.max(requested_ks)}")
    
        # 构建指令包
        payloads = []
        for i in range(self.num_envs):
            data = {
                'chunk': self.last_chunks[i],
                'k': requested_ks[i]
            }
            # [修改] 如果有分布数据则发送，训练时通常为 None
            if self.next_step_dists is not None:
                data['dist_probs'] = self.next_step_dists[i]
            
            payloads.append(data)

        # 清空
        self.next_step_dists = None
        
        self.actions_cache = requested_ks
        # self.last_chunks = batch_chunks # 存下来，等 step_wait 时更新历史
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
            # 获取实际执行了多少步
            k_executed = infos_all[i].get("exec_len", 0)
            
            # --- 【核心】：在观测返回前更新动作历史 ---
            if k_executed > 0:
                # 取得该环境上一轮执行的那些动作
                actions_taken = self.last_chunks[i][:k_executed]
                self._update_action_history(i, actions_taken)
            
            # 如果环境 Reset 了，历史清零
            if dones_all[i]:
                self.action_histories[i] = 0.0

            # 处理重采样惩罚 (actions_cache 在 step_async 中保存)
            if self.actions_cache[i] == 0:
                combined_rewards[i] = self.resample_penalty
            
        # 3. 推理下一阶段计划并构建下一帧 RL 观测
        new_plans = self._get_batch_policy_plan(self.current_obs_dicts)
        self.last_chunks = new_plans # 更新缓存
        
        next_rl_obs = np.stack([
            self._make_rl_obs(i, {k: v[i] for k, v in self.current_obs_dicts.items()}, new_plans[i]) 
            for i in range(self.num_envs)
        ])
        
        return next_rl_obs, combined_rewards, dones_all, infos_all

    def reset(self):
        # 1. 获取 Batch 字典 {'key': [N, ...]}
        obs_batch = self.venv.reset()
        self.current_obs_dicts = obs_batch

        # Reset 时历史全部清零
        self.action_histories.fill(0.0)
        
        # 2. 一次性推理出所有环境的计划
        batch_chunks = self._get_batch_policy_plan(self.current_obs_dicts)
        self.last_chunks = batch_chunks

        # 4. 构造完整的 RL 观测 [Obs, History, Chunk]
        # 我们先单独算一下第一个环境的，用来做探针演示
        sample_rl_obs_list = []
        for i in range(self.num_envs):
            single_obs_dict = {k: v[i] for k, v in self.current_obs_dicts.items()}
            rl_obs = self._make_rl_obs(i, single_obs_dict, batch_chunks[i])
            sample_rl_obs_list.append(rl_obs)
        
        final_obs_stack = np.stack(sample_rl_obs_list)

        # =========================================================================
        # 【历史验证探针】
        # =========================================================================
        # 动态获取物理维度：通过计算单个环境的物理向量
        sample_phys_vec = self._get_obs_vec({k: v[0] for k, v in obs_batch.items()})
        phys_dim = sample_phys_vec.shape[0]
        hist_dim = self.action_history_len * self.action_dim
        
        # 提取第一个环境观测向量中的“历史切片”
        # 顺序是 [Physical(0:phys_dim), History(phys_dim:phys_dim+hist_dim), Chunk(...)]
        history_slice = final_obs_stack[0, phys_dim : phys_dim + hist_dim]
        
        # 计算理论上的“零动作归一化值”
        # 因为物理上是填0，所以归一化后应该是 (0 - mean) / std
        expected_norm_zero = -self.stats["action_mean"] / self.stats["action_std"]
        theoretical_mean = np.mean(expected_norm_zero)
        actual_mean = np.mean(history_slice)

        return final_obs_stack

        # # 3. 将 Batch 字典拆分并为每个环境构建 RL 观测
        # return np.stack([
        #     self._make_rl_obs(i, {k: v[i] for k, v in self.current_obs_dicts.items()}, batch_chunks[i]) 
        #     for i in range(self.num_envs)
        # ])

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

    # =========================================================================
    # 【新增方法】用于处理 Timeout 时的 Terminal Observation
    # =========================================================================
    def get_terminal_rl_obs(self, terminal_obs_dict, env_idx):
        """
        严谨版：处理超时瞬间的观测转换，包含该环境正确的动作历史。
        """
        # 1. 构造 Batch 并推理 (Batch Size = 1)
        obs_batch_for_dp = {}
        valid_keys = list(self.algo_instance.obs_shapes.keys())
        for k in valid_keys:
            if k in terminal_obs_dict:
                val = np.array(terminal_obs_dict[k])
                if val.ndim == 1: val = val[None, :]
                obs_batch_for_dp[k] = val
        
        # 2. 推理新的 Chunk
        chunk = self._get_batch_policy_plan(obs_batch_for_dp)[0]
        
        # 3. 拼接 RL 向量 (传入 env_idx 以获取该环境的 history)
        return self._make_rl_obs(env_idx, terminal_obs_dict, chunk)

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