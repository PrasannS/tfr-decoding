# SFT MODEL 
#python merge_peft_adapter.py --adapter_model_name="trl-lib/llama-7b-se-peft" --base_model_name="/home/prasann/Projects/tfr-decoding/llama/llama" --output_name="./models/"
# RM 
#python merge_peft_adapter.py --adapter_model_name="trl-lib/llama-7b-se-rm-peft" --base_model_name="/home/prasann/Projects/tfr-decoding/llama/llama" --output_name="./models/rewardmodel/"
# SFT MODEL attempt #2 (first one seemed too good, I suspect they accidentally uploaded the better one?)
python merge_peft_adapter.py --adapter_model_name="mnoukhov/llama-7b-se-peft" --base_model_name="/home/prasann/Projects/tfr-decoding/llama/llama" --output_name="./models/"
