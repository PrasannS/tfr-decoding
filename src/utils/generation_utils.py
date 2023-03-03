import logging
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BartForConditionalGeneration, T5ForConditionalGeneration
import pandas as pd
import torch
from src.tfr_decoding.custom_bs import beam_search
from src.models.models import load_from_checkpoint as lfc
import time

def load_model(setting, tfrdecode=True, device="cuda:1", train=False):
    if setting == "noun":
        logging.info('Loading xsum model')
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
    if setting == "table2text":
        usebart=False
        if usebart:
            tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
        else:
            tokenizer = AutoTokenizer.from_pretrained("t5-large")

        new_tokens = ['<H>', '<R>', '<T>']
        new_tokens_vocab = {}
        new_tokens_vocab['additional_special_tokens'] = []
        for idx, t in enumerate(new_tokens):
            new_tokens_vocab['additional_special_tokens'].append(t)
        num_added_toks = tokenizer.add_special_tokens(new_tokens_vocab)
        # first get cond gen model
        if usebart:
            ckpt = torch.load("/mnt/data1/prasann/latticegen/lattice-generation/parent_explore/plms-graph2text/webnlg-bart-base.ckpt")
        else:
            ckpt = torch.load("/mnt/data1/prasann/latticegen/lattice-generation/parent_explore/plms-graph2text/webnlg-t5-large.ckpt")          
        state_dict = ckpt['state_dict']
        # make weight keys compatible 
        for key in list(state_dict.keys()):
            if key[:len("model.")]=="model.":
                state_dict[key[len("model."):]] = state_dict.pop(key)
        if usebart:
            model = BartForConditionalGeneration.from_pretrained(
                "facebook/bart-base", state_dict=ckpt['state_dict'], vocab_size=50268
            ).to(device)
        else:
            model = T5ForConditionalGeneration.from_pretrained(
                "t5-large", state_dict=ckpt['state_dict'], vocab_size=tokenizer.vocab_size+num_added_toks
            ).to(device)
        model.eval()
        # dataset to train stuff
        if train:
            datadf = pd.read_csv("/mnt/data1/prasann/tfr-decoding/webnlg_train.csv")
            tlines = list(datadf['reference'])
        else:
            datadf = pd.read_csv("/mnt/data1/prasann/latticegen/lattice-generation/parent_explore/stagewise_finetune/parent_master/wnlg_testset_bart.csv")
            tlines = list(datadf['ref'])
        slines = list(datadf['src'])
        
        dataset = zip(slines, tlines)
        dec_prefix = [model.config.decoder_start_token_id]
    
    # do necessary monkey patching
    if tfrdecode:
        model.beam_search = beam_search.__get__(model)
        tfrmodel = lfc(setting, True).to(device)
        tfrmodel.eval()
        model.tfr = tfrmodel
        model.tokenizer = tokenizer
        model.tfr_tok = model.tfr.encoder.tokenizer
        model.config.forced_bos_token_id = None
        # TODO add more depending on the setting
        model.no_source = setting in ["noun"]
        
    return model, tokenizer, dataset, dec_prefix

def generate_cands(mod, tok, src, args):
    inps = tok([src], return_tensors="pt", truncation=True).to(args["device"])
    outputs = mod.generate(**inps, max_new_tokens=args["max_len"], 
            return_dict_in_generate=True, output_scores=True,
            num_beams=args["beam_size"], num_return_sequences=args["beam_size"])
    return tok.batch_decode(outputs.sequences, skip_special_tokens=True), outputs.sequences_scores

# TODO incorporate source for source input settings (anything not NOUN-TFR)
def tfr_decode_ind(md, tk, src, args):
    md.tfr_interv = args['tfr_interv']
    md.tfr_beams = args['tfr_beams']
    md.weightfunc = args['weightfunc']
    md.source_str = src # TODO do something less janky
    preds, scos = generate_cands(md, tk, src, args)
    
    return preds, scos

LOGSTEPS = 20
def all_tfr_decode(mod, tok, dset, args):
    res = []
    ind = 1
    start = time.time()
    for ex in dset:
        if ind%LOGSTEPS==0:
            print(ind)
        #try:
        cands, scores = tfr_decode_ind(mod, tok, ex[0], args)
        #except:
            #print("decoding failed")
        for c in range(len(scores)):
            res.append({
                'ref':ex[1],
                'hyp':cands[c],
                'src':ex[0],
                'modsco':scores[c]
            })
        ind+=1
    timetot = time.time()-start
    print("Time taken : ", timetot)
    return pd.DataFrame(res)