#accelerate launch --multi_gpu --num_machines 1  --num_processes 4 \
# switch to single gpu since we don't seem to gain anything from multi-gpu
export CUDA_VISIBLE_DEVICES=1
export WORLD_SIZE=3
accelerate launch --num_machines 1 --num_processes 1 --config_file=custom_config.yaml \
    tuning_lm_with_rl.py \
    --log_with wandb \
    --adafactor False \
    --save_freq 30 \
    --output_max_length 200 \
    --gradient_accumulation_steps 4 \
    --mini_batch_size 8 \
    --batch_size 16 \
    --batched_gen True \
    --ppo_epochs 10 \
    --learning_rate 1.4e-5 \
    --early_stopping False \
    --output_dir './checkpoints/llama_oasst_v3/'

# NOTE: if we're using a custom algo, 
# - batch_size needs to be adjusted to hardcoded constraint
# - adjust hardcoded constraint in file
# - make sure file is good
# - make sure output_dir is correct
# - make sure we have dedicated device
# - distinct output file, don't override useful info

# v2 is longer seq lengths with 0.01 KL
# v3 is longer seq lengths with 0.1 KL