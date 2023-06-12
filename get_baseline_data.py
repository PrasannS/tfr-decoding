from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from datasets import load_dataset
import pandas as pd
from readability import Readability
import numpy as np
from src.tfr_decoding.custom_bs import beam_search
from src.tfr_decoding.recurse_samp import sample

device = 'cuda:3' # if you have a GPU

# get generation model
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map="auto")#.to(device)
model.beam_search = beam_search.__get__(model)
model.sample = sample.__get__(model)
model.eval()

# get shp model
steamtok = T5Tokenizer.from_pretrained('stanfordnlp/SteamSHP-flan-t5-xl')
steamshp = T5ForConditionalGeneration.from_pretrained('stanfordnlp/SteamSHP-flan-t5-xl').to(device)

eli5 = load_dataset("stanfordnlp/shp", data_dir="explainlikeimfive")
eliorig = pd.DataFrame(eli5['test'])
elidf = eliorig.drop_duplicates(subset="history")

# make prompt for eli5
def construct_prompt(row):
    otemplate = \
"""
The system will write a detailed and long post to respond to the user's question. Explain like the user is five years old. 

Question: """
    template = \
"""
Give a lengthy, detailed response. 

Question: """
    inp = template+row['history']+"\n Detailed Response:"
    
    
    return inp

# score a single example (I don't think there's enough space to batch this?)
def get_reward_single(inpdict):
    template = "POST: {context:s} \n\nRESPONSE A:{hyp:s} \n\nRESPONSE B: .\n\n Which response is better? RESPONSE "
    inp = template.format(context=inpdict['context'], hyp=inpdict['hyp'])
    x = steamtok([inp], return_tensors='pt').input_ids.to(device)
    outputs = steamshp.generate(x, return_dict_in_generate=True, output_scores=True, max_new_tokens=1)
    return torch.exp(outputs.scores[0][:, 71]) / torch.exp(outputs.scores[0][:,:]).sum(axis=1).item() # index 71 corresponds to the token for 'A'

# generate output for an input row
def gen_row(rw, tok, mod, method="greedy", num_hyps=10):
    input_text = construct_prompt(rw)
    
    #print(input_text)
    input_ids = tok(input_text, return_tensors="pt").input_ids.to(device)
    if method=="greedy":
        outputs = mod.generate(input_ids, min_new_tokens=20, max_new_tokens=200)
        outs = [tok.decode(outputs[0], skip_special_tokens=True)]
    elif method=="sample": 
        outputs = mod.generate(input_ids, min_new_tokens=20, max_new_tokens=200, do_sample=True, top_p=.95, temperature=.9, num_return_sequences=num_hyps)
        outs = [tok.decode(o, skip_special_tokens=True) for o in outputs]
    elif method=="beam":
        outputs = mod.generate(input_ids, min_new_tokens=20, max_new_tokens=200, num_beams=num_hyps, num_return_sequences=num_hyps)
        outs = [tok.decode(o, skip_special_tokens=True) for o in outputs]

    return rw['history'], outs

def gen_dir_beam(rw, tok, mod, pfs, keepR, newBeams, ssamps):
    model.man_pref=None
    # generate with initial sample
    inp, outs = gen_row(rw, tok, mod, "sample", ssamps)
    # generate scores to re-rank, only use best options for next step
    shp_scores = [float(get_reward_single({"context": inp, "hyp":o})) for o in outs]
    bestopts = list(np.argsort(shp_scores))
    bestopts.reverse()
    bestopts = bestopts[:keepR]
    nouts = ["<pad> "+outs[bo].strip() for bo in bestopts]
    spls = [nout.split() for nout in nouts]
    spls = [sp[:int(len(sp)*pfs[i])] for sp in spls]
    splens = [len(l) for l in spls]

    # cut based on average of decoded seq lens
    prefcut = int(int(sum(splens)/len(splens))*pfs[i])
    # TODO setup for multiple inputs
    forced = tok(nouts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)[:, :-1][:, :prefcut]
    print("PREFIX: ")
    print(tok.batch_decode(forced))
    mod.man_pref = forced
    
    inp, dirouts = gen_row(rw, tok, mod, "beam", newBeams)
    
    dir_scos = [float(get_reward_single({"context": inp, "hyp":o})) for o in dirouts]
    
    return inp, outs, shp_scores, dirouts, dir_scos

def printout(p, oros, orscs, os, scs):
    print("PROMPT: ")
    print(p)
    print("ORIGINAL SAMPS: ")
    for s in list(zip(orscs, oros)):
        print(s)
    print("NEW: ")
    for s in list(zip(scs, os)):
        print(s)


def gen_rec_samp(rw, tok, mod, pfs, keepR, tsamps):
    model.man_pref=None
    # generate with initial sample
    interm_hyps = []
    interm_scos = []
    for i in range(len(keepR)):
        inp, outs = gen_row(rw, tok, mod, "sample", tsamps[i])
        # generate scores to re-rank, only use best options for next step
        shp_scores = [float(get_reward_single({"context": inp, "hyp":o})) for o in outs]
        interm_hyps.append(outs)
        interm_scos.append(shp_scores)
        # sort by shp scores
        bestopts = list(np.argsort(shp_scores))
        bestopts.reverse()
        bestopts = bestopts[:keepR[i]]
        nouts = ["<pad>"+outs[bo].strip() for bo in bestopts]
        # calculate prefix lens for prefcut calculation
        spls = [nout.split() for nout in nouts]
        #spls = [sp[:int(len(sp)*pfs[i])] for sp in spls]
        splens = [len(l) for l in spls]
        print(splens)
        
        prefcut = int(int(sum(splens)/len(splens))*pfs[i])
        print("prefcut ", prefcut)
        # TODO setup for multiple inputs
        forced = tok(nouts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)[:, :-1]
        forced = forced[:, :prefcut]
        if i==len(tsamps)-1:
            tlen = tsamps[i]
        else:
            tlen = tsamps[i+1]
        forced = forced.repeat(int(tlen/keepR[i]), 1)
        print("PREFIX: ")
        print(tok.batch_decode(forced))
        if pfs[i]>0:
            mod.man_pref = forced
    
    return inp, interm_hyps, interm_scos, outs, shp_scores

def samprecind(ind, pflen, rchoose, tsamps, log=False):
    # algorithm 2
   
    prompt, orig_os, orig_scos, outs, scos = gen_rec_samp(elidf.iloc[ind], tokenizer, model, pflen, rchoose, tsamps)
    if log:
        print("Prefix len ", pflen, "; Chosen Top-N ", rchoose)
        printout(prompt, orig_os, orig_scos, outs, scos)
    return prompt, orig_os, orig_scos, outs, scos

def sampbase(ind, log=False):
    # algorithm 2
    pflen = [.1, .1, .1]
    rchoose = [2, 2, 2]
    tsamps = [4, 4,4]
    prompt, orig_os, orig_scos, outs, scos = gen_rec_samp(elidf.iloc[ind], tokenizer, model, pflen, rchoose, tsamps)
    if log:
        print("Prefix len ", pflen, "; Chosen Top-N ", rchoose)
        printout(prompt, orig_os, orig_scos, outs, scos)
    return prompt, orig_os, orig_scos, outs, scos

def gendirind(ind, log=False):
    pflen = .125
    startsamps = 8
    rchoose = 2
    tsamps = 2
    prompt, orig_os, orig_scos, outs, scos = gen_dir_beam(elidf.iloc[ind], tokenizer, model, pflen, rchoose, tsamps, startsamps)
    if log:
        print("Prefix len ", pflen, "; Chosen Top-N ", rchoose)
        printout(prompt, orig_os, orig_scos, outs, scos)
    return prompt, orig_os, orig_scos, outs, scos

def samprecall(tot):
    res = []
    for i in range(tot):
        print(i)
        prompt, orig_os, orig_scos, outs, scos = samprecind(i)
        # store all data
        res.append({
            'input':prompt,
            'allhyps':orig_os,
            'allscos':orig_scos
        })
    return res
    
def samprecind(ind, pflen, rchoose, tsamps, log=False):
    # algorithm 2
   
    prompt, orig_os, orig_scos, outs, scos = gen_rec_samp(elidf.iloc[ind], tokenizer, model, pflen, rchoose, tsamps)
    if log:
        print("Prefix len ", pflen, "; Chosen Top-N ", rchoose)
        printout(prompt, orig_os, orig_scos, outs, scos)
    return prompt, orig_os, orig_scos, outs, scos

def sampbase(ind, log=False):
    # algorithm 2
    pflen = [.1, .1, .1]
    rchoose = [2, 2, 2]
    tsamps = [4, 4,4]
    prompt, orig_os, orig_scos, outs, scos = gen_rec_samp(elidf.iloc[ind], tokenizer, model, pflen, rchoose, tsamps)
    if log:
        print("Prefix len ", pflen, "; Chosen Top-N ", rchoose)
        printout(prompt, orig_os, orig_scos, outs, scos)
    return prompt, orig_os, orig_scos, outs, scos

def gendirind(ind, log=False):
    pflen = .125
    startsamps = 8
    rchoose = 2
    tsamps = 2
    prompt, orig_os, orig_scos, outs, scos = gen_dir_beam(elidf.iloc[ind], tokenizer, model, pflen, rchoose, tsamps, startsamps)
    if log:
        print("Prefix len ", pflen, "; Chosen Top-N ", rchoose)
        printout(prompt, orig_os, orig_scos, outs, scos)
    return prompt, orig_os, orig_scos, outs, scos

def samprecall(start, end, pfls, rchs, tsmps):
    res = []
    for i in range(start, end):
        print(i)
        prompt, orig_os, orig_scos, outs, scos = samprecind(i, pfls, rchs, tsmps)
        # store all data
        res.append({
            'input':prompt,
            'allhyps':orig_os,
            'allscos':orig_scos
        })
    return pd.DataFrame(res)
    
pflen = [0, 0, 0, 0, 0]
rchoose = [2, 2, 2, 2, 2]
tsamps = [6, 6, 6, 6, 6]
#samprecind(2, pflen, rchoose, tsamps)

df1 = samprecall(0,50, pflen, rchoose, tsamps)
df1.to_json("baselines1.jsonl", orient="records", lines=True)
df2 = samprecall(51,100, pflen, rchoose, tsamps)
df2.to_json("baselines2.jsonl", orient="records", lines=True)