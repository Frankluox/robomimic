CUDA_VISIBLE_DEVICES=1 python /home/wuhao/jobspace/robomimic/robomimic/scripts/train_adaptive_parallel.py \
    --agent /home/wuhao/jobspace/robomimic.wh/wuhao/ckpt/dp/low_dim_tool_hang/20260123043153/models/model_epoch_12.pth \
    --max_chunk_len 32 \
    --reward_offset 1.0 \
    --resample_penalty -5.0 \
    --total_timesteps 200000 \
    --n_steps 256 \
    --n_epochs 10 \
    --n_envs 8 \
    --batch_size 512 \
    --log_dir /home/wuhao/jobspace/robomimic/wuhao/adaptive_rl_results/tb_logs/standard_brandnew_nonorm \
    --ckpt_dir /home/wuhao/jobspace/robomimic/wuhao/adaptive_rl_results/checkpoints/standard_brandnew_nonorm \
    --save_freq 20000 \
    --save_videos

    # --lr 5e-5 \

    # --log_dir /home/wuhao/jobspace/robomimic/wuhao/adaptive_rl_results/tb_logs/standard_parallel_v1 \
    # --ckpt_dir /home/wuhao/jobspace/robomimic/wuhao/adaptive_rl_results/checkpoints/standard_parallel_v1 \