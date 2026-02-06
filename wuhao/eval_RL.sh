#!/bin/bash

# Base Diffusion Policy (冻结的底层策略)
AGENT="/home/wuhao/jobspace/luoxu/robomimic/models/tool_hang_epoch_12.pth"

# RL Model
RL_MODEL="/home/wuhao/jobspace/luoxu/robomimic/wuhao/adaptive_rl_results/tb_logs/new_server_optimized/PPO_0131_170419/checkpoints/adaptive_agent_4999200_steps.zip"

# 基础输出目录
# BASE_OUTPUT_DIR="/home/wuhao/jobspace/luoxu/robomimic/wuhao/adaptive_rl_results/eval_results/eval_adaptive_parallel_new_server_optimized_4999200_deterministic"
# BASE_OUTPUT_DIR="/home/wuhao/jobspace/luoxu/robomimic/wuhao/adaptive_rl_results/eval_results/fixed9_again"
BASE_OUTPUT_DIR="/home/wuhao/jobspace/luoxu/robomimic/wuhao/adaptive_rl_results/eval_results/save_data_test"

# 定义 3 个不同的种子
# SEEDS=(2024 2025 2026)
SEEDS=(2025)

# 循环测试
for SEED in "${SEEDS[@]}"; do
    echo "========================================"
    echo "正在测试 Seed: ${SEED}"
    echo "========================================"
    
    CURR_DIR="${BASE_OUTPUT_DIR}/seed_${SEED}"
    
    (
        CUDA_VISIBLE_DEVICES=3 python /home/wuhao/jobspace/luoxu/robomimic/robomimic/scripts/evaluate_adaptive_parallel.py \
            --agent ${AGENT} \
            --n_rollouts 5 \
            --rl_model ${RL_MODEL} \
            --n_envs 2 \
            --horizon 600 \
            --seed ${SEED} \
            --max_chunk_len 32 \
            --action_history_len 0 \
            --video_dir ${CURR_DIR} \
            --save_data \
            # --fixed_chunk_len 9  # 如果需要测固定长度，取消注释上面并注释掉 --rl_model
                        # --rl_model ${RL_MODEL} \
            # --save_video \
            # --rl_model ${RL_MODEL} \
    ) &
done

# 重要：等待所有后台进程结束
echo "[INFO] All seeds launched. Waiting for them to finish..."
wait
echo "[INFO] All evaluation processes completed."

# --- 汇总逻辑：计算平均值、标准差并保存到 summary.json ---
echo "========================================"
echo "最终汇总结果 (Statistics over ${#SEEDS[@]} seeds):"
python -c "
import json, os, numpy as np
base_dir = '${BASE_OUTPUT_DIR}'
# [核心修复] 使用 split() 将 Bash 传入的空格分隔字符串转为 Python 整数列表
seeds = [int(s) for s in '${SEEDS[*]}'.split()]
results = []

for s in seeds:
    path = os.path.join(base_dir, f'seed_{s}', 'eval_results.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            results.append(json.load(f))

if results:
    # 提取数据
    srs = [r['success_rate'] for r in results]
    rets = [r['avg_return'] for r in results]
    
    # 计算统计量
    mean_sr = np.mean(srs) * 100
    std_sr = np.std(srs) * 100
    mean_ret = np.mean(rets)
    std_ret = np.std(rets)
    
    # 打印到控制台
    print(f'测试种子: {seeds}')
    print(f'各种子成功率: {[round(x*100, 2) for x in srs]} %')
    print('-' * 40)
    print(f'平均成功率 (Success Rate): {mean_sr:.2f} ± {std_sr:.2f} %')
    print(f'平均奖励值 (Average Return): {mean_ret:.4f} ± {std_ret:.4f}')
    print('=' * 40)

    # --- 新增：保存到 summary.json ---
    summary_data = {
        'seeds': seeds,
        'num_seeds': len(results),
        'raw_data': {
            'success_rates': srs,
            'returns': rets
        },
        'statistics': {
            'mean_success_rate': float(mean_sr),
            'std_success_rate': float(std_sr),
            'mean_return': float(mean_ret),
            'std_return': float(std_ret)
        }
    }
    
    summary_path = os.path.join(base_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=4)
    print(f'[INFO] Final summary saved to: {summary_path}')
    
else:
    print('错误：未找到任何结果文件，请检查评估是否正常完成。')
"