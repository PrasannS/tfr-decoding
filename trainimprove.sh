export CUDA_VISIBLE_DEVICES=1,2,3  # Use the first two GPUs
#nohup python -u train_improves.py > trainim.out &
nohup python -u train_prefmod_pairwise.py > compmodel.out &
#nohup python -u train_prefmod_contrastive.py > contpmod2.out &
