from alpaca_farm.models import reward_model
from alpaca_farm.inference.decode import load_model_and_tokenizer_for_inference 
from alpaca_farm import utils
import alpaca_farm.data_preprocessor as data_preprocessor

import torch
import pandas as pd

from src.utils.ap_samp_utils import exhaustive_samp

def set_w_prompts(indf):
    instrs = []
    pdict_path = "/home/prasann/Projects/tfr-decoding/alpaca_farm/examples/prompts/v0_inputs_noinputs.json"
    for i, r in indf.iterrows():
        prompts, list_dict_data, metadata = data_preprocessor.format_prompt_with_data_frame(
            df=pd.DataFrame({'instruction':[r['history']], 'input':[""]}),
            prompt_dict=utils.jload(pdict_path),
        )
        instrs.append(prompts[0])
    return instrs

if __name__=="__main__":
    elidf = pd.read_json("output/elidataset.jsonl", orient="records", lines="true")
    
    # do this for set of 100 examples
    elidf = elidf.drop_duplicates(subset="history").iloc[15000:15100]   
    elidf['history'] = set_w_prompts(elidf)

    rew_model, tokenizer = load_model_and_tokenizer_for_inference(
        model_name_or_path="/home/prasann/Projects/tfr-decoding/apfarm_models/reward-model-human/",
        model_cls=reward_model.RewardModel,
        cache_dir=None,
        model_kwargs=dict(
            torch_dtype=utils.convert_str_dtype_to_torch_dtype(None),
            flash_attn=False,
        ),
    )
    # load in ppo generations
    # model, tokenizer = load_model_and_tokenizer_for_inference(
    #     model_name_or_path='/home/prasann/Projects/tfr-decoding/apfarm_models/ppo-human/',
    #     cache_dir=None,
    #     model_kwargs=dict(torch_dtype=utils.convert_str_dtype_to_torch_dtype(None)),
    # )
    # model.eval()
    
    # exppo = exhaustive_samp("ppoexhaust", elidf, tokenizer, model, tokenizer, rew_model, [4, 4], [4, 4], 1)
    
    # del model
    # del tokenizer
    torch.cuda.empty_cache()
    
    # load in sft generations
    model, tokenizer = load_model_and_tokenizer_for_inference(
        model_name_or_path='/home/prasann/Projects/tfr-decoding/apfarm_models/sft10k/',
        cache_dir=None,
        model_kwargs=dict(torch_dtype=utils.convert_str_dtype_to_torch_dtype(None)),
    )
    model.eval()
    
    exsft = exhaustive_samp("sftexhaust", elidf, tokenizer, model, tokenizer, rew_model, [4, 4], [4, 4], 1)