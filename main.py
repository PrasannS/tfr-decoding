from src.tfr_decoding.tfr_decode import run_baseline_decoding
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.tfr_decoding.custom_bs import beam_search
import logging
import pandas as pd

def load_model(setting, device="cuda:0"):
    if setting == "xsum":
        logging.info('Loading model')
        # load up model
        model_name = 'facebook/bart-large-xsum'
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        logging.info('Loading dataset')
        
        # TODO I am hard-coding this
        datadf = pd.read_csv("/mnt/data1/prasann/latticegen/lattice-generation/mt-data/summarytestset.csv")
        slines = list(datadf['src'])
        tlines = list(datadf['ref'])
        dataset = zip(slines, tlines)
        dec_prefix = [tokenizer.eos_token_id] # TODO Jiacheng had this as BOS
    return model, tokenizer, dataset, dec_prefix

def run_hf_baseline(mod, tok, src, args, custom=False):
    inps = tok([src], return_tensors="pt").to(args["device"])
    outputs = mod.generate(**inps, max_new_tokens=args["max_len"], 
            return_dict_in_generate=True, output_scores=True,
            num_beams=args["beam_size"], num_return_sequences=args["beam_size"])
    return tok.batch_decode(outputs.sequences)
    

if __name__=="__main__":
    # get model, tokenizer
    mod, tok, dset, dec_pref = load_model("xsum", "cuda:0")

    # manually override to insert our beam_search method 
    mod.beam_search = beam_search.__get__(mod)

    # TODO will need to insert TFR model for use as well

    # have some input string
    source, ref = list(dset)[0]
    print("SRC: ", source)
    print("REF: ", ref)
    args = {
        "max_len":90,
        "device":'cuda:0',
        "beam_size":2,
        "dec_prefix":dec_pref
    }

    # run algorithm / cross fingers
    base = run_hf_baseline(mod, tok, source, args, False)
    # our goal is for this to match
    preds = run_hf_baseline(mod, tok, source, args, True)
    
    print(preds)