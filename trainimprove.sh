export CUDA_VISIBLE_DEVICES=2,3  # Use the first two GPUs
#nohup python -u train_improves.py > trainim.out &
nohup python -u explore_pfsamp.py > hypexp2.out &
