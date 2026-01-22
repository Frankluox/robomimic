"""
The main script for evaluating a policy in an environment.

Args:
    agent (str): path to saved checkpoint pth file

    horizon (int): if provided, override maximum horizon of rollout from the one 
        in the checkpoint

    env (str): if provided, override name of env from the one in the checkpoint,
        and use it for rollouts

    render (bool): if flag is provided, use on-screen rendering during rollouts

    video_path (str): if provided, render trajectories to this video file path

    video_skip (int): render frames to a video every @video_skip steps

    camera_names (str or [str]): camera name(s) to use for rendering on-screen or to video

    dataset_path (str): if provided, an hdf5 file will be written at this path with the
        rollout data

    dataset_obs (bool): if flag is provided, and @dataset_path is provided, include 
        possible high-dimensional observations in output dataset hdf5 file (by default,
        observations are excluded and only simulator states are saved).

    seed (int): if provided, set seed for rollouts

Example usage:

    # Evaluate a policy with 50 rollouts of maximum horizon 400 and save the rollouts to a video.
    # Visualize the agentview and wrist cameras during the rollout.
    
    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --video_path /path/to/output.mp4 \
        --camera_names agentview robot0_eye_in_hand 

    # Write the 50 agent rollouts to a new dataset hdf5.

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5 --dataset_obs 

    # Write the 50 agent rollouts to a new dataset hdf5, but exclude the dataset observations
    # since they might be high-dimensional (they can be extracted again using the
    # dataset_states_to_obs.py script).

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5
"""
import argparse
import json
import h5py
import imageio
import numpy as np
from copy import deepcopy
import os
import gc
import multiprocessing
import threading  # [新增] 用于监听日志队列
import queue      # [新增]
import pickle # [新增] 用于保存数据
import sys # [新增] 用于控制标准输出

import torch

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy


# [新增] 定义一个只写文件的 Logger，强制刷新以防丢失日志
class FileLogger(object):
    def __init__(self, filename, mode="a"):
        self.log = open(filename, mode)
    
    def write(self, message):
        self.log.write(message)
        self.log.flush() # [关键] 强制刷新，确保内容立即写入硬盘
        
    def flush(self):
        self.log.flush()
        
    def close(self):
        self.log.close()

# [新增] 监听线程函数：在主进程中运行，负责打印各 Worker 的状态
def log_listener(msg_queue):
    while True:
        try:
            msg = msg_queue.get()
            if msg == "KILL": # 结束信号
                break
            print(msg) # 这里是主进程，打印到真正的终端
            sys.stdout.flush()
        except Exception:
            break


def rollout(policy, env, horizon, render=False, video_writer=None, video_skip=5, return_obs=False, camera_names=None, rollout_id=0, log_dir=None, init_state=None):
    """
    Helper function to carry out rollouts. Supports on-screen rendering, off-screen rendering to a video, 
    and returns the rollout trajectory.

    Args:
        policy (instance of RolloutPolicy): policy loaded from a checkpoint
        env (instance of EnvBase): env loaded from a checkpoint or demonstration metadata
        horizon (int): maximum horizon for the rollout
        render (bool): whether to render rollout on-screen
        video_writer (imageio writer): if provided, use to write rollout to video
        video_skip (int): how often to write video frames
        return_obs (bool): if True, return possibly high-dimensional observations along the trajectoryu. 
            They are excluded by default because the low-dimensional simulation states should be a minimal 
            representation of the environment. 
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.

    Returns:
        stats (dict): some statistics for the rollout - such as return, horizon, and task success
        traj (dict): dictionary that corresponds to the rollout trajectory
    """
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))


    # [新增] 初始化数据存储列表
    # 我们将存储 (step, image, full_chunk, truncation_length)


    evaluation_logs = []


    

    policy.start_episode()

    
    if init_state is not None:
        obs = env.reset_to(init_state)
        state_dict = init_state
    else:
        obs = env.reset()
        state_dict = env.get_state()
        # hack that is necessary for robosuite tasks for deterministic action playback
        obs = env.reset_to(state_dict)

    results = {}
    video_count = 0  # video frame counter
    total_reward = 0.
    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)
    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))

    success = False # 初始化 success 变量，防止 horizon=0 时报错

    try:
        for step_i in range(horizon):


            # get action from policy
            act = policy(ob=obs)

            # print(act)

            # [新增] --- 数据记录逻辑开始 ---
            # 1. 获取图像
            # 需要 env.render (注意这可能会影响性能)
            image_to_log = None
            # print(obs.keys())

            try:
                # 无论 render 是否为 True，只要 render_offscreen 开启了就能渲染
                image_to_log = env.render(
                    mode="rgb_array", 
                    height=128, 
                    width=128, 
                    camera_name=camera_names[0]
                )
            except Exception as e:
                # 只有在第一步打印错误，避免日志刷屏
                if step_i == 0:
                    print(f"渲染失败，请检查是否在 eval.sh 中设置了 --video_path: {e}")

            # 2. 获取 Policy 内部的日志
            # policy 是 RolloutPolicy 实例，policy.policy 是 DiffusionPolicyUNet 实例
            internal_policy = policy.policy 
            if hasattr(internal_policy, "last_execution_info") and internal_policy.last_execution_info:
                info = internal_policy.last_execution_info
                
                # 我们只在发生推理的那一步记录（或者你想每步都记也可以，但数据量会很大）
                # 这里假设只要有 info 且是新推理的(step_log=True)就记录
                if info.get("step_log", True):
                    log_entry = {
                        "step": step_i,
                        "image": image_to_log, # 注意：图片数据量很大，确保存储空间足够
                        "full_action_chunk": info["full_action_chunk"],
                        "truncation_length": info["final_len"],
                        "use_sched": info["use_sched"],
                        "use_ood": info["use_ood"],
                        "ood_score": info["ood_score"],
                        "reason": info["reason"],
                        "ood_suggested_bound": info["ood_suggested_bound"],
                    }
                    evaluation_logs.append(log_entry)
            # [新增] --- 数据记录逻辑结束 ---
            
            # play action
            next_obs, r, done, _ = env.step(act)

            # compute reward
            total_reward += r
            success = env.is_success()["task"]

            # visualization
            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                    video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                    video_writer.append_data(video_img)
                video_count += 1

            # collect transition
            traj["actions"].append(act)
            traj["rewards"].append(r)
            traj["dones"].append(done)
            traj["states"].append(state_dict["states"])
            if return_obs:
                traj["obs"].append(obs)
                traj["next_obs"].append(next_obs)

            # break if done or if success
            if done or success:
                break

            # update for next iter
            obs = deepcopy(next_obs)
            state_dict = env.get_state()

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    # 这里处理 step_i 还没定义的情况 (如 horizon=0 或直接报错)
    horizon_len = step_i + 1 if 'step_i' in locals() else 0
    stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))


    # [新增] --- 保存日志 ---
    # 建议保存为 pickle 文件，文件名包含一些唯一标识
    # 注意：由于是在 rollout 函数里，可能需要传入额外的参数来指定保存路径或 ID
    # 这里简单起见，假设你可以在 args 里传个路径前缀
    if len(evaluation_logs) > 0 and log_dir:
        save_path = os.path.join(log_dir, f"log_rollout_{rollout_id}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(evaluation_logs, f)

    # [修改] 使用传入的 log_dir 保存日志
    if len(evaluation_logs) > 0 and log_dir:
        # 确保目录存在 (虽然主进程已经创建，但双重保险无害)
        os.makedirs(log_dir, exist_ok=True)
        save_path = os.path.join(log_dir, f"log_rollout_{rollout_id}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(evaluation_logs, f)
            print(f"Saved rollout log to {save_path}") # 这行会被重定向

    if return_obs:
        # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
        traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
        traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return stats, traj


def rollout_parallel_wrapper(args_tuple):
    """
    用于多进程的 Worker 函数。
    """

    ckpt_path, env_name, horizon, seed, device_str, dataset_obs, camera_names, video_path, video_skip, index, log_dir, msg_queue, action_horizon, use_action_scheduler, use_ood_monitor, init_state = args_tuple

    

    # [新增] 设置日志重定向

    # [关键修改] 每个 Worker 无条件重定向到自己的日志文件
    # 不再判断 index == 0
    if log_dir is not None:
        # 每个进程独立的文本日志
        log_file = os.path.join(log_dir, f"worker_rollout_{index}.txt")
        # 保存原始 stdout 以便恢复(可选，但在 Pool 中通常不需要)

        original_stdout = sys.stdout 
        original_stderr = sys.stderr
        
        file_logger = FileLogger(log_file)
        sys.stdout = file_logger
        sys.stderr = file_logger

    try:
        # 向主进程报告：开始
        if msg_queue:
            msg_queue.put(f"[Main] Task {index} Started on PID {os.getpid()}...")

        device = torch.device(device_str)
        
        # 此时所有的 print 都会进入 log_file
        print(f"Start processing rollout {index} on device {device}")

        # 加载策略 (verbose=False 减少日志输出)
        policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=False)

        # [NEW] Inject the flags into the underlying policy model
        # policy is a RolloutPolicy wrapper, policy.policy is the DiffusionPolicyUNet instance
        if hasattr(policy.policy, "use_action_scheduler"):
            policy.policy.use_action_scheduler = use_action_scheduler
            if msg_queue and index == 0:
                msg_queue.put(f"Set use_action_scheduler to {use_action_scheduler}")

        if hasattr(policy.policy, "use_ood_monitor"):
            policy.policy.use_ood_monitor = use_ood_monitor
            if msg_queue and index == 0:
                msg_queue.put(f"Set use_ood_monitor to {use_ood_monitor}")

        # [关键修改]：如果有传入 action_horizon，强制覆盖 policy 内部的配置
        if action_horizon is not None:
            # policy 是 RolloutPolicy，policy.policy 才是实际的算法实例(DiffusionPolicyUNet)
            # 必须同时修改 algo_config，因为 reset() 函数也会用到它来设置队列长度
            with policy.policy.algo_config.values_unlocked():
                policy.policy.algo_config.horizon.action_horizon = action_horizon
            # 建议打印一下确认覆盖成功 (只在第0个worker打印避免刷屏)
            if msg_queue:
                msg_queue.put(f"Overriding Action Horizon to: {action_horizon}")
                print(f"Overriding Action Horizon to: {action_horizon}")

        # 创建独立的环境实例
        # 这里的 render 必须设为 False，因为多进程无法抢占屏幕渲染
        env, _ = FileUtils.env_from_checkpoint(
            ckpt_dict=ckpt_dict, 
            env_name=env_name, 
            render=False, 
            render_offscreen=(video_path is not None), 
            verbose=False,
        )

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        video_writer = None
        if video_path is not None:
            # 构造唯一文件名，例如: output.mp4 -> output_0.mp4
            root, ext = os.path.splitext(video_path)
            worker_video_path = f"{root}_{index}{ext}"
            try:
                video_writer = imageio.get_writer(worker_video_path, fps=20)
            except Exception as e:
                print(f"Error creating video writer: {e}")

        stats, traj = rollout(
            policy=policy, 
            env=env, 
            horizon=horizon, 
            render=False, 
            video_writer=video_writer, 
            video_skip=video_skip, 
            return_obs=dataset_obs,
            camera_names=camera_names,
            rollout_id=index,
            # log_dir="/home/wuhao/jobspace/robomimic/wuhao/logs" # 传入保存目录
            log_dir=log_dir, # 传入保存目录
            init_state=init_state
        )
    
        # [修改] 视频重命名逻辑 (Success / Failed)
        if video_path is not None and worker_video_path is not None:
            try:
                video_writer.close() # 显式关闭以防万一
            except:
                pass
                
            try:
                if stats['Success_Rate'] > 0:
                    new_name = worker_video_path.replace(".mp4", "_success.mp4")
                    os.rename(worker_video_path, new_name)
                    print(f"Video saved to {new_name}")
                else:
                    # [新增] 失败也改名
                    new_name = worker_video_path.replace(".mp4", "_failed.mp4")
                    os.rename(worker_video_path, new_name)
                    print(f"Video saved to {new_name}")
            except OSError as e:
                print(f"Error renaming video file: {e}")

        # 向主进程报告：结束

        status_str = "SUCCESS" if stats['Success_Rate'] > 0 else "FAILED"
        if msg_queue:
            msg_queue.put(f"[Main] Task {index} Finished: {status_str} (Return: {stats['Return']:.2f})")

        del policy  
        del env     
        gc.collect() 
        torch.cuda.empty_cache() 
    

        return stats, traj

    except Exception as e:
        # 捕获所有异常并打印到日志文件，同时通知主进程
        print(f"CRITICAL ERROR in Worker {index}: {e}")
        import traceback
        traceback.print_exc()
        if msg_queue:
            msg_queue.put(f"[Main] Task {index} CRASHED! Check log: worker_rollout_{index}.txt")
        return None, None
    finally:
        # [可选] 可以在这里关闭 file_logger，虽然进程结束会自动关闭
        if 'file_logger' in locals():
            file_logger.close()


def run_trained_agent(args):

    write_video = (args.video_path is not None)
    assert not (args.render and write_video) # either on-screen or video but not both
    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.camera_names) == 1

    # relative path to agent
    ckpt_path = args.agent
    
    if args.device == "cpu":
        device = "cpu"
    else:
        device = "cuda"
        multiprocessing.set_start_method('spawn') # 下面多进程fork模式会报错


    # [新增] 自动创建 Video 目录
    if args.video_path:
        video_dir = os.path.dirname(args.video_path)
        if video_dir and not os.path.exists(video_dir):
            os.makedirs(video_dir, exist_ok=True)
            print(f"Created video directory: {video_dir}")

    # [新增] 自动创建 Log 目录
    if args.log_dir:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir, exist_ok=True)
            print(f"Created log directory: {args.log_dir}")

    # restore policy (只用来读 horizon，不用于多进程推理)
    _, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=False)

    # read rollout settings
    rollout_num_episodes = args.n_rollouts
    rollout_horizon = args.horizon
    if rollout_horizon is None:
        # read horizon from config
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        rollout_horizon = config.experiment.rollout.horizon

    # create environment from saved checkpoint
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict, 
        env_name=args.env, 
        render=args.render, 
        render_offscreen=(args.video_path is not None), 
        verbose=True,
    )

    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # maybe open hdf5 to write rollouts
    write_dataset = (args.dataset_path is not None)
    if write_dataset:
        # [新增] 自动创建 Dataset 目录
        dataset_dir = os.path.dirname(args.dataset_path)
        if dataset_dir and not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir, exist_ok=True)

        data_writer = h5py.File(args.dataset_path, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0



    # ============== 修改核心部分 =================

    # [新增] 使用 Manager().Queue() 进行跨进程通信
    manager = multiprocessing.Manager()
    msg_queue = manager.Queue()

    # [新增] 启动后台线程监听并打印 Queue 里的消息
    listener = threading.Thread(target=log_listener, args=(msg_queue,))
    listener.daemon = True # 设置为守护线程，主程序退出时自动退出
    listener.start()

    # results = []

    # device = cuda，需要考虑显存是否足够大？多进程的启动必须用spawn模式，防止继承父进程的上下文导致cuda报错
    # device = cpu，需要考虑内存是否足够大？model是否小到cpu也可以推理？多进程的启动可以用fork模式 启动会快一点
    num_workers = min(args.parallel, args.n_rollouts) 
    print(f"Running evaluation with {num_workers} processes on {args.device.upper()}...")
    print(f"Logs for each rollout will be saved in {args.log_dir}")

    init_states = None
    if args.init_states_file:
        print(f"Loading initial states from {args.init_states_file}...")
        with open(args.init_states_file, 'rb') as f:
            init_states = pickle.load(f)
        print(f"Loaded {len(init_states)} states.")
        assert len(init_states) >= args.n_rollouts, "Not enough initial states provided."

    work_items = []
    
    for i in range(rollout_num_episodes):
        curr_seed = args.seed + i if args.seed is not None else None

        init_state = None
        if init_states is not None:
            init_state = init_states[i]

        work_items.append((
            ckpt_path, 
            args.env, 
            rollout_horizon, 
            curr_seed, 
            device,
            (write_dataset and args.dataset_obs),
            args.camera_names,
            args.video_path, 
            args.video_skip,
            i,
            args.log_dir,# [新增] 传入 log_dir          
            msg_queue, # [新增] 传入 queue    
            args.action_horizon, # [新增] 将命令行参数传入 Worker 
            args.use_action_scheduler, # [NEW] Pass argument
            args.use_ood_monitor,       # [NEW] Pass argument
            init_state       
        ))

    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(rollout_parallel_wrapper, work_items)

    # 发送结束信号给监听线程（或者直接让它随主进程销毁）
    msg_queue.put("KILL")
    listener.join()


    rollout_stats = []
    # 过滤掉 None 结果 (crash 的任务)
    valid_results = [r for r in results if r[0] is not None]

    for i, (stats, traj) in enumerate(results):
        rollout_stats.append(stats)
        if write_dataset:
            ep_data_grp = data_grp.create_group("demo_{}".format(i))
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            if args.dataset_obs:
                for k in traj["obs"]:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
                    ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

            if "model" in traj["initial_state_dict"]:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"]
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0]
            total_samples += traj["actions"].shape[0]

    # ============== 修改结束 =================

    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    # [修改] 将 numpy 类型转换为原生 Python 类型，确保 JSON 序列化不出错
    avg_rollout_stats = { k : float(np.mean(rollout_stats[k])) for k in rollout_stats }
    avg_rollout_stats["Num_Success"] = int(np.sum(rollout_stats["Success_Rate"]))

    print("Average Rollout Stats")
    print(json.dumps(avg_rollout_stats, indent=4))

    # [新增] 将结果保存到指定的 log_dir 目录下
    if args.log_dir is not None:
        # 确保目录存在
        os.makedirs(args.log_dir, exist_ok=True)
        result_json_path = os.path.join(args.log_dir, "results.json")
        with open(result_json_path, 'w') as f:
            json.dump(avg_rollout_stats, f, indent=4)
        print(f"Saved average rollout stats to {result_json_path}")


    if write_dataset:
        # 需要重新加载 env 才能 serialize 吗？
        # env 变量在主进程中被释放了吗？如果不确定，可以重新加载一个 dummy env
        # 或者在 worker 返回时带上 env info
        # 这里假设 args.env 可用
        try:
             # 为了获取 env info，快速加载一个 dummy
            dummy_env, _ = FileUtils.env_from_checkpoint(
                ckpt_dict=ckpt_dict, env_name=args.env, render=False, verbose=False
            )
            data_grp.attrs["total"] = total_samples
            data_grp.attrs["env_args"] = json.dumps(dummy_env.serialize(), indent=4)
        except Exception as e:
            print(f"Warning: Could not save env args to dataset: {e}")

        data_writer.close()
        print("Wrote dataset trajectories to {}".format(args.dataset_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to trained model
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="path to saved checkpoint pth file",
    )

    # number of rollouts
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=50,
        help="number of rollouts",
    )

    # maximum horizon of rollout, to override the one stored in the model checkpoint
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="(optional) override maximum horizon of rollout from the one in the checkpoint",
    )

    # Env Name (to override the one stored in model checkpoint)
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="(optional) override name of env from the one in the checkpoint, and use\
            it for rollouts",
    )

    # Whether to render rollouts to screen
    parser.add_argument(
        "--render",
        action='store_true',
        help="on-screen rendering",
    )

    # Dump a video of the rollouts to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render rollouts to this video file path",
    )

    # How often to write video frames during the rollout
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # camera names to render
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=["agentview"],
        help="(optional) camera name(s) to use for rendering on-screen or to video",
    )

    # If provided, an hdf5 file will be written with the rollout data
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="(optional) if provided, an hdf5 file will be written at this path with the rollout data",
    )

    # If True and @dataset_path is supplied, will write possibly high-dimensional observations to dataset.
    parser.add_argument(
        "--dataset_obs",
        action='store_true',
        help="include possibly high-dimensional observations in output dataset hdf5 file (by default,\
            observations are excluded and only simulator states are saved)",
    )

    # for seeding before starting rollouts
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(optional) set seed for rollouts",
    )

    # added bu wuhao
    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=4
    )

    # [新增] action_horizon 参数
    parser.add_argument(
        "--action_horizon",
        type=int,
        default=None,
        help="(optional) override action execution horizon (Ta) defined in config",
    )

    # [NEW] Add these lines
    parser.add_argument(
        "--use_action_scheduler", 
        action='store_true', 
        help="enable action scheduler in Diffusion Policy"
    )
    parser.add_argument(
        "--use_ood_monitor", 
        action='store_true', 
        help="enable OOD monitor in Diffusion Policy"
    )

    parser.add_argument(
        "--init_states_file",
        type=str,
        default=None,
        help="path to pickled initial states file for deterministic evaluation",
    )

    # [新增] log_dir 参数
    parser.add_argument("--log_dir", type=str, default=None, help="directory to save log files (pkl and txt)")

    args = parser.parse_args()
    run_trained_agent(args)

