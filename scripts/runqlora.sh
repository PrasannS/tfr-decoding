export CUDA_VISIBLE_DEVICES=1,2,3
python -u qlora/qlora.py --model_name_or_path "/home/prasann/Projects/tfr-decoding//llama/llama" --dataset="/home/prasann/Projects/tfr-decoding//output/llamaftunetrain.jsonl"
