# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1
#nohup python -u prefix_sampling.py > psamp.out &
nohup python -u test_pref_hyp.py > psamp4.out &