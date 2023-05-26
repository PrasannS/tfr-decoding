export CUDA_VISIBLE_DEVICES=0,1  # Use the first two GPUs
#nohup python -u train_improves.py > trainim.out &
#nohup python -u explore_pfsamp.py > hypexp3.out &
nohup python -u train_prefmod_contrastive.py > multicont.out &
