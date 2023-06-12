# get necessary models from alpacafarm
python -m pretrained_models.recover_model_weights \
  --llama-7b-hf-dir "/mnt/data1/prasann/prefixdecoding/tfr-decoding/llama/llama" \
  --alpaca-farm-model-name ppo-human \
  --models-save-dir "../apfarm_models"

python -m pretrained_models.recover_model_weights \
  --llama-7b-hf-dir "/mnt/data1/prasann/prefixdecoding/tfr-decoding/llama/llama" \
  --alpaca-farm-model-name reward-model-human \
  --models-save-dir "../apfarm_models"