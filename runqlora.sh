# export CUDA_VISIBLE_DEVICES=
#python -u qlora/qlora.py --model_name_or_path "/mnt/data1/prasann/prefixdecoding/tfr-decoding/llama/llama" --dataset="/mnt/data1/prasann/prefixdecoding/tfr-decoding/output/llamaftunetrain.jsonl"

# GET WEIGHTS FOR ALPACAFARM

python -m pretrained_models.recover_model_weights \
  --llama-7b-hf-dir <your_path_to_hf_converted_llama_ckpt_and_tokenizer> \
  --alpaca-farm-model-name <one_of_the_model_names_from_above> \
  --models-save-dir <dir_to_save_all_models>