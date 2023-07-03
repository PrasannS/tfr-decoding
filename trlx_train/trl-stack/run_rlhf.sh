export CUDA_VISIBLE_DEVICES=2,3
export WORLD_SIZE=3
accelerate launch --multi_gpu --config_file=/home/prasann/Projects/tfr-decoding/trlx_train/trl-stack/default_config.yaml \
    --num_machines 1  \
    --num_processes 2 \
    rl_training.py --log_with=wandb \
    --model_name=/home/prasann/Projects/tfr-decoding/trlx_train/trl-stack/models/sft \
    --reward_model_name=/home/prasann/Projects/tfr-decoding/trlx_train/trl-stack/models/rewardmodel \
    --adafactor=False \
    --tokenizer_name=/home/prasann/Projects/tfr-decoding/trlx_train/trl-stack/models/sft \
    --save_freq=100 \
    --output_max_length=128 --batch_size=8 \
    --gradient_accumulation_steps=32 --batched_gen=True \
    --ppo_epochs=4 --seed=0 --learning_rate=1.4e-5 \
    --early_stopping=False --output_dir=checkpoints/v2/