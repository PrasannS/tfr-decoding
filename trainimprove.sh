export CUDA_VISIBLE_DEVICES=1 # Use the first two GPUs
#nohup python -u train_improves.py > trainim.out &
nohup python -u train_prefmod_multi.py > ap_pref_ppo.out &
#nohup python -u train_prefmod_contrastive.py > contpmod2.out &
