# square 0.64
# CUDA_VISIBLE_DEVICES=0 \
# python /home/wuhao/jobspace/robomimic/robomimic/scripts/run_trained_agent.py \
# --agent /home/wuhao/jobspace/robomimic/wuhao/ckpt/dp/low_dim_square/20260121020517/models/model_epoch_2_low_dim_v15_success_0.64.pth \
# --n_rollouts 50 --horizon 400 --seed 0 --video_path /home/wuhao/jobspace/robomimic/wuhao/video/square_0.64/output.mp4 \
# --camera_names agentview robot0_eye_in_hand --device cuda --parallel 8

# square 0.44
# CUDA_VISIBLE_DEVICES=0 \
# python /home/wuhao/jobspace/robomimic/robomimic/scripts/run_trained_agent.py \
# --agent /home/wuhao/jobspace/robomimic/wuhao/ckpt/dp/low_dim_square/20260121020517/models/model_epoch_10.pth \
# --n_rollouts 50 --horizon 400 --seed 0 --video_path /home/wuhao/jobspace/robomimic/wuhao/video/square_0.44/output.mp4 \
# --camera_names agentview robot0_eye_in_hand --device cuda --parallel 8

# square 0.64 f
# CUDA_VISIBLE_DEVICES=0 \
# python /home/wuhao/jobspace/robomimic/robomimic/scripts/run_trained_agent.py \
# --agent /home/wuhao/jobspace/robomimic/wuhao/ckpt/dp/low_dim_square/20260121020517/models/model_epoch_2_low_dim_v15_success_0.64.pth \
# --n_rollouts 50 --horizon 400 --seed 0 --video_path /home/wuhao/jobspace/robomimic/wuhao/video/square_0.64_f/output.mp4 \
# --camera_names agentview robot0_eye_in_hand --device cuda --parallel 8

# square 0.44 f
# CUDA_VISIBLE_DEVICES=0 \
# python /home/wuhao/jobspace/robomimic/robomimic/scripts/run_trained_agent.py \
# --agent /home/wuhao/jobspace/robomimic/wuhao/ckpt/dp/low_dim_square/20260121020517/models/model_epoch_10.pth \
# --n_rollouts 50 --horizon 400 --seed 0 --video_path /home/wuhao/jobspace/robomimic/wuhao/video/square_0.44_f/output.mp4 \
# --camera_names agentview robot0_eye_in_hand --device cuda --parallel 8

# tool hang 0.18
# CUDA_VISIBLE_DEVICES=1 \
# python /home/wuhao/jobspace/robomimic/robomimic/scripts/run_trained_agent.py \
# --agent /home/wuhao/jobspace/robomimic/wuhao/ckpt/dp/low_dim_tool_hang/20260121084406/models/model_epoch_10_low_dim_v15_success_0.18.pth \
# --n_rollouts 50 --horizon 600 --seed 0 --video_path /home/wuhao/jobspace/robomimic/wuhao/video/tool_hang/output.mp4 \
# --camera_names agentview robot0_eye_in_hand --device cuda --parallel 6

# tool hang 0.18 f
# CUDA_VISIBLE_DEVICES=1 \
# python /home/wuhao/jobspace/robomimic/robomimic/scripts/run_trained_agent.py \
# --agent /home/wuhao/jobspace/robomimic/wuhao/ckpt/dp/low_dim_tool_hang/20260121084406/models/model_epoch_10_low_dim_v15_success_0.18.pth \
# --n_rollouts 50 --horizon 600 --seed 0 --video_path /home/wuhao/jobspace/robomimic/wuhao/video/tool_hang_f/output.mp4 \
# --camera_names agentview robot0_eye_in_hand --device cuda --parallel 6

EXP_NAME="baseline_10"


CUDA_VISIBLE_DEVICES=0 \
python /home/wuhao/jobspace/robomimic/robomimic/scripts/run_trained_agent.py \
--agent /home/wuhao/jobspace/robomimic/wuhao/ckpt/dp/low_dim_square/20260121020517/models/model_epoch_2_low_dim_v15_success_0.64.pth \
--n_rollouts 50 --horizon 400 --seed 0 --device cuda --parallel 8 --video_path /home/wuhao/jobspace/robomimic/wuhao/video/${EXP_NAME}/output.mp4 \
--camera_names agentview --log_dir /home/wuhao/jobspace/robomimic/wuhao/logs/${EXP_NAME} --action_horizon 10 \
# --use_action_scheduler --use_ood_monitor


# CUDA_VISIBLE_DEVICES=0 \
# python /home/wuhao/jobspace/robomimic/robomimic/scripts/run_trained_agent_wh.py \
# --agent /home/wuhao/jobspace/robomimic/wuhao/ckpt/dp/low_dim_square/20260121020517/models/model_epoch_2_low_dim_v15_success_0.64.pth \
# --n_rollouts 50 --horizon 400 --seed 0 --device cuda --parallel 8 