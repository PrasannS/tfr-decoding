export CUDA_VISIBLE_DEVICES=2,3  # Use the first two GPUs
#nohup python -u trainprefquality.py > trainq.out &
nohup python -u test_pref_hyp.py > gettestset.out &