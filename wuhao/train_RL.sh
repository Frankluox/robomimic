CUDA_VISIBLE_DEVICES=1 python /home/wuhao/jobspace/robomimic/robomimic/scripts/train_adaptive.py \
    --agent /home/wuhao/jobspace/robomimic/wuhao/ckpt/dp/low_dim_tool_hang/20260121084406/models/model_epoch_10_low_dim_v15_success_0.18.pth \
    --max_chunk_len 16 \
    --reward_offset 0.005 \
    --resample_penalty -0.01 \
    --total_timesteps 1000000 \
    --n_steps 256 \
    --n_envs 8 \
    --log_dir /home/wuhao/jobspace/robomimic/wuhao/adaptive_rl_results/tb_logs/all_obs_test \
    --ckpt_dir /home/wuhao/jobspace/robomimic/wuhao/adaptive_rl_results/checkpoints/all_obs_test