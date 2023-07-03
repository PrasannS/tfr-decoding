import torch
import numpy as np
import pandas as pd
from sigfig import round as rnd
import random
from alpaca_farm.inference.score import score_sequences_with_huggingface_given_model

def get_reward_single(seq, rmtok, rm):
    # TODO make a batched version if necessary
    return score_sequences_with_huggingface_given_model(rm, rmtok, [seq], per_device_batch_size=1)[0]

# generate output for an input row
def gen_row(rw, tok, mod, method="greedy", num_hyps=10, temp=.9, mintoks=20):
    #input_text = construct_prompt(rw)
    # assume it's already given in prompt form
    if mod.man_pref is not None:
        input_ids = mod.man_pref
    else:
        input_text = rw['history']
        input_ids = tok(input_text, return_tensors="pt").input_ids.to(mod.device)
    print("Inputs: ")
    print(tok.batch_decode(input_ids))
    if method=="sample": 
        # NOTE removed min_new_tokens
        outputs = mod.generate(input_ids, max_new_tokens=200, do_sample=True, top_p=.95, temperature=temp, num_return_sequences=int(num_hyps/len(input_ids)), return_dict_in_generate=True, output_scores=True)
        outs = [tok.decode(o, skip_special_tokens=True) for o in outputs.sequences]
    print("Outputs: \n", outs)
    return rw['history'], outs, outputs.scores

def gen_rec_samp(rw, tok, mod, stmtok, stmshp, pfs, keepR, tsamps, startinp=None, srow=-1, temp=.9):
    mod.man_pref=None
    # generate with initial sample
    interm_hyps = []
    interm_scos = []
    for i in range(len(keepR)):
        if startinp is not None and i==0:
            outs = startinp["allhyps"][srow]
            shp_scores = startinp['allscos'][srow]
        else:
            inp, outs, _ = gen_row(rw, tok, mod, "sample", tsamps[i], temp)
            # generate scores to re-rank, only use best options for next step
            shp_scores = [float(get_reward_single(o, stmtok, stmshp)) for o in outs]
            
        interm_hyps.append(outs)
        interm_scos.append(shp_scores)
        # sort by shp scores
        bestopts = list(np.argsort(shp_scores))
        bestopts.reverse()
        bestopts = bestopts[:keepR[i]]
        # NOTE removed <pad> token
        nouts = [outs[bo].strip() for bo in bestopts]
        # calculate prefix lens for prefcut calculation
        spls = [nout.split() for nout in nouts]
        #spls = [sp[:int(len(sp)*pfs[i])] for sp in spls]
        splens = [len(l) for l in spls]
        #print(splens)
        # option for manual token
        if pfs[i]>1:
            prefcut=pfs[i]
        else:
            prefcut = int(int(sum(splens)/len(splens))*pfs[i])
        # print("prefcut ", prefcut)
        # TODO setup for multiple inputs
        forced = tok(nouts, return_tensors="pt", padding=True, truncation=True).input_ids.to(mod.device)[:, :-1]
        # NOTE get rid of </s> at beginning
        forced = forced[:, 1:prefcut]
        if i==len(tsamps)-1:
            tlen = tsamps[i]
        else:
            tlen = tsamps[i+1]
        forced = forced.repeat(int(tlen/keepR[i]), 1)
        #print("PREFIX: ")
        #print(tok.batch_decode(forced))
        if pfs[i]>0:
            mod.man_pref = forced
    
    return inp, interm_hyps, interm_scos, outs, shp_scores, None

def sampfrominp(datadf, tok, mod, stmtok, stmshp, ind, row, inps, pflen, rchoose, tsamps, temp):
    if inps:
        prompt, orig_os, orig_scos, outs, scos, statdicts = gen_rec_samp(datadf.iloc[ind], tok, mod, stmtok, stmshp, pflen, rchoose, tsamps, inps, row, temp)
    else:
        prompt, orig_os, orig_scos, outs, scos, statdicts = gen_rec_samp(datadf.iloc[ind], tok, mod, stmtok, stmshp, pflen, rchoose, tsamps, None, row, temp)
    nb = max(orig_scos[-1])
    ob = max(orig_scos[0])
    nav = mean(orig_scos[-1])
    oav = mean(orig_scos[0])
    if inps:
        oldscos = inps['allscos']
        obests = list([max(m) for m in oldscos])
        oavgs = list([sum(m)/len(m) for m in oldscos])
    
    #print("newbest ", nb, "; oldbest ", ob)
    print("newavg ", nav, "; oldavg", oav)
    if inps:
        #print("manybest ", mean(obests))
        #print("manyavg", mean(oavgs))

        return prompt, orig_os, orig_scos, nav, oav, obests, oavgs, nb, ob, statdicts
    return prompt, orig_os, orig_scos, nav, oav, None, None, nb, ob, statdicts

def exhaustive_samp(savefile, datadf, tok, mod, stmtok, stmshp, rchoose, tsamps, temp):
    allvals = []
    for i in range(len(datadf)):
        # get initial sample to go off of
        prompt, orig_os, orig_scos, nav, oav, obests, oavgs, nb, ob, stats = sampfrominp(datadf, tok, mod, stmtok, stmshp, i, 0, None, [0.5], rchoose[:1], tsamps[:1], temp)
        allvals.append({
            'inp':prompt,
            'hyps':orig_os[0],
            'scos':orig_scos[0],
            "pref":0
        })
        # index of prompt
        start = len(tok(prompt).input_ids)
        # take len of best path in initial sample, we'll resample several times from here
        end = len(tok(orig_os[0][0]).input_ids) 
        inps = {
            'allhyps':[[orig_os[0][0]]*tsamps[0]],
            'allscos':[orig_scos[0]]
        }
        # allow for resampling up to 120 tokens (40 resamps per example)
        for j in range(start+3, min(end, start+40*3)):
            try:
                # resample every 3 tokens
                if (j-start)%3==0:
                    prompt, orig_os, orig_scos, nav, oav, obests, oavgs, nb, ob, stats = sampfrominp(datadf, tok, mod, stmtok, stmshp, i, 0, inps, [j, -1], rchoose, tsamps, temp)
                    #print(orig_os[1])
                    allvals.append({
                        'inp':prompt,
                        'hyps':orig_os[1],
                        'scos':orig_scos[1],
                        "prefix":j
                    })
                # save progress every 10
                if (len(allvals)+1)%10==0:
                    tmp = pd.DataFrame(allvals)
                    tmp.to_json("output/apexhaust/"+savefile+".jsonl", orient="records", lines=True)
            except:
                print("something went wrong")
                torch.cuda.empty_cache()
    tmp = pd.DataFrame(allvals)
    tmp.to_json("output/apexhaust/"+savefile+".jsonl", orient="records", lines=True)          
    return tmp

def mean(l):
    return sum(l)/len(l)
    