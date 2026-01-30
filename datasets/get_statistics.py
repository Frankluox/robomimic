import h5py
import numpy as np
import os

def calculate_dataset_stats(dataset_path, output_path="toolhang_stats.npz"):
    # 需要统计的 Observation Key
    obs_keys = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object']
    
    # 存储数据的容器
    obs_data = {k: [] for k in obs_keys}
    action_data = []
    
    print(f"正在读取数据集: {dataset_path}")
    
    with h5py.File(dataset_path, "r") as f:
        demos = list(f["data"].keys())
        num_demos = len(demos)
        
        for i, demo_key in enumerate(demos):
            # 1. 提取 Observation 数据
            obs_grp = f[f"data/{demo_key}/obs"]
            for k in obs_keys:
                if k in obs_grp:
                    obs_data[k].append(obs_grp[k][()])
            
            # 2. 提取 Action 数据 (Shape: [T, action_dim])
            if "actions" in f[f"data/{demo_key}"]:
                action_data.append(f[f"data/{demo_key}/actions"][()])

            if (i + 1) % 50 == 0:
                print(f"已处理 {i + 1}/{num_demos} 条轨迹...")

    # 用于保存最终结果的字典
    save_dict = {}

    # --- 计算 Observation 统计量 ---
    print("\n正在计算 Observation 统计量...")
    for k in obs_keys:
        if len(obs_data[k]) > 0:
            all_obs = np.concatenate(obs_data[k], axis=0)
            save_dict[f"{k}_mean"] = np.mean(all_obs, axis=0)
            save_dict[f"{k}_std"] = np.std(all_obs, axis=0) + 1e-6 # 防止除零
            print(f"  [Obs] {k:20} | Shape: {save_dict[f'{k}_mean'].shape}")

    # --- 计算 Action 统计量 ---
    print("正在计算 Action 统计量...")
    if len(action_data) > 0:
        all_actions = np.concatenate(action_data, axis=0)
        # print(f"  总动作数据形状: {all_actions.shape}")
        save_dict["action_mean"] = np.mean(all_actions, axis=0)
        save_dict["action_std"] = np.std(all_actions, axis=0) + 1e-6
        print(f"  [Action] {'action':17} | Shape: {save_dict['action_mean'].shape}")

    # 保存文件
    np.savez(output_path, **save_dict)
    print(f"\n统计量已成功保存至: {output_path}")

# 执行
dataset_path = "/home/wuhao/jobspace/robomimic/datasets/tool_hang/ph/low_dim_v15.hdf5"
calculate_dataset_stats(dataset_path, "/home/wuhao/jobspace/robomimic/datasets/tool_hang/ph/toolhang_stats.npz")