CUDA_VISIBLE_DEVICES=2 python /home/wuhao/jobspace/robomimic/robomimic/scripts/train_adaptive_parallel.py \
    --agent /home/wuhao/jobspace/robomimic.wh/wuhao/ckpt/dp/low_dim_tool_hang/20260123043153/models/model_epoch_12.pth \
    --max_chunk_len 32 \
    --reward_offset 1.0 \
    --resample_penalty -5.0 \
    --total_timesteps 1000000 \
    --n_steps 256 \
    --n_epochs 10 \
    --n_envs 8 \
    --batch_size 512 \
    --log_dir /home/wuhao/jobspace/robomimic/wuhao/adaptive_rl_results/tb_logs/1.30_standard_lr_1e-5 \
    --save_freq 20000 \
    --gamma 0.999 \
    --gae_lambda 0.99 \
    --lr 1e-5 \
    --action_history_len 0

    # --lr 5e-5 \
        # --save_videos \

    # --log_dir /home/wuhao/jobspace/robomimic/wuhao/adaptive_rl_results/tb_logs/standard_parallel_v1 \
    # --ckpt_dir /home/wuhao/jobspace/robomimic/wuhao/adaptive_rl_results/checkpoints/standard_parallel_v1 \