import torch
import numpy as np
import pandas as pd
from sigfig import round as rnd
import random


# make prompt for eli5
def construct_prompt(row):
    template = \
"""
Give a lengthy, detailed response. 
Question: """
    inp = template+row['history']+"\n Detailed Response:"
    
    
    return inp

# score a single example (I don't think there's enough space to batch this?)
def get_reward_single(inpdict, steamtok, steamshp):
    template = "POST: {context:s} \n\nRESPONSE A:{hyp:s} \n\nRESPONSE B: .\n\n Which response is better? RESPONSE "
    inp = template.format(context=inpdict['context'], hyp=inpdict['hyp'])
    x = steamtok([inp], return_tensors='pt').input_ids.to(steamshp.device)
    outputs = steamshp.generate(x, return_dict_in_generate=True, output_scores=True, max_new_tokens=1)
    return torch.exp(outputs.scores[0][:, 71]) / torch.exp(outputs.scores[0][:,:]).sum(axis=1).item() # index 71 corresponds to the token for 'A'

# generate output for an input row
def gen_row(rw, tok, mod, method="greedy", num_hyps=10, temp=.9):
    input_text = construct_prompt(rw)
    
    #print(input_text)
    input_ids = tok(input_text, return_tensors="pt").input_ids.to(mod.device)
    if method=="greedy":
        outputs = mod.generate(input_ids, min_new_tokens=20, max_new_tokens=200)
        outs = [tok.decode(outputs[0], skip_special_tokens=True)]
    elif method=="sample": 
        outputs = mod.generate(input_ids, min_new_tokens=20, max_new_tokens=300, do_sample=True, top_p=.95, temperature=temp, num_return_sequences=num_hyps, return_dict_in_generate=True, output_scores=True)
        outs = [tok.decode(o, skip_special_tokens=True) for o in outputs.sequences]
    elif method=="beam":
        outputs = mod.generate(input_ids, min_new_tokens=20, max_new_tokens=200, num_beams=num_hyps, num_return_sequences=num_hyps)
        outs = [tok.decode(o, skip_special_tokens=True) for o in outputs]

    return rw['history'], outs, outputs.scores

def gen_dir_beam(rw, tok, mod, pfs, keepR, newBeams, ssamps):
    mod.man_pref=None
    # generate with initial sample
    inp, outs, scoredf = gen_row(rw, tok, mod, "sample", ssamps)
    # generate scores to re-rank, only use best options for next step
    shp_scores = [float(get_reward_single({"context": inp, "hyp":o})) for o in outs]
    bestopts = list(np.argsort(shp_scores))
    bestopts.reverse()
    bestopts = bestopts[:keepR]
    nouts = ["<pad> "+outs[bo].strip() for bo in bestopts]
    spls = [nout.split() for nout in nouts]
    spls = [sp[:int(len(sp)*pfs[0])] for sp in spls]
    splens = [len(l) for l in spls]

    # cut based on average of decoded seq lens
    prefcut = int(int(sum(splens)/len(splens))*pfs[0])
    # TODO setup for multiple inputs
    forced = tok(nouts, return_tensors="pt", padding=True, truncation=True).input_ids.to(mod.device)[:, :-1][:, :prefcut]
    print("PREFIX: ")
    print(tok.batch_decode(forced))
    mod.man_pref = forced
    
    inp, dirouts = gen_row(rw, tok, mod, "beam", newBeams)
    
    dir_scos = [float(get_reward_single({"context": inp, "hyp":o})) for o in dirouts]
    
    return inp, outs, shp_scores, dirouts, dir_scos, scoredf

def printout(p, oros, orscs, os, scs):
    print("PROMPT: ")
    print(p)
    print("ORIGINAL SAMPS: ")
    for s in list(zip(orscs, oros)):
        print(s)
    print("NEW: ")
    for s in list(zip(scs, os)):
        print(s)


def gen_rec_samp(rw, tok, mod, stmtok, stmshp, pfs, keepR, tsamps, startinp=None, srow=-1, temp=.9):
    mod.man_pref=None
    # generate with initial sample
    interm_hyps = []
    interm_scos = []
    scodfs = []
    for i in range(len(keepR)):
        if startinp is not None and i==0:
            outs = startinp["allhyps"][srow]
            shp_scores = startinp['allscos'][srow]
            scoredf = None
        else:
            inp, outs, scoredf = gen_row(rw, tok, mod, "sample", tsamps[i], temp)
            # generate scores to re-rank, only use best options for next step
            shp_scores = [float(get_reward_single({"context": inp, "hyp":o}, stmtok, stmshp)) for o in outs]
            for k in scoredf.keys():
                # convert to floats for easier storage
                scoredf[k] = [[rnd(c.item(), 4) for c in row] for row in scoredf[k]]
        interm_hyps.append(outs)
        interm_scos.append(shp_scores)
        scodfs.append(scoredf)
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
        # option for manual token
        if pfs[i]>1:
            prefcut=pfs[i]
        else:
            prefcut = int(int(sum(splens)/len(splens))*pfs[i])
        print("prefcut ", prefcut)
        # TODO setup for multiple inputs
        forced = tok(nouts, return_tensors="pt", padding=True, truncation=True).input_ids.to(mod.device)[:, :-1]
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
    
    return inp, interm_hyps, interm_scos, outs, shp_scores, scodfs

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
    
    print("newbest ", nb, "; oldbest ", ob)
    print("newavg ", nav, "; oldavg", oav)
    if inps:
        print("manybest ", mean(obests))
        print("manyavg", mean(oavgs))

        return prompt, orig_os, orig_scos, nav, oav, obests, oavgs, nb, ob, statdicts
    return prompt, orig_os, orig_scos, nav, oav, None, None, nb, ob, statdicts

def inpsampall(datadf, tok, mod, stmtok, stmshp, inps, pflen, rchoose, tsamps, numind, temp):
    allvals = []
    for i in range(len(datadf)):
        prompt, orig_os, orig_scos, nav, oav, obests, oavgs, nb, ob, stats = sampfrominp(datadf, tok, mod, stmtok, stmshp, i, numind, inps, pflen, rchoose, tsamps, temp)
        allvals.append({
            'inp':prompt,
            'hyps':orig_os,
            'scos':orig_scos,
            'new_avg':nav,
            "old_avg":oav,
            "newmax":nb,
            "oldmax":ob,
            "obests":obests,
            "oavgs":oavgs,
            "stats":stats
        })
        tmp = pd.DataFrame(allvals)
        tmp.to_json("output/prefpreds/verbprefpreds"+str(numind)+"_"+str(pflen[0])+".jsonl", lines=True, orient="records")
    return tmp

def exhaustive_samp(datadf, tok, mod, stmtok, stmshp, rchoose, tsamps, temp):
    allvals = []
    for i in range(len(datadf)):
        # get initial sample to go off of
        prompt, orig_os, orig_scos, nav, oav, obests, oavgs, nb, ob, stats = sampfrominp(datadf, tok, mod, stmtok, stmshp, i, 0, None, [0.5, 0], rchoose, tsamps, temp)
        allvals.append({
            'inp':prompt,
            'hyps':orig_os[0],
            'scos':orig_scos[0],
            "stats":stats[0],
            "ver":"first",
            "pref":0
        })
        bscoind = np.argmax(orig_scos[0])
        # take len of best path in initial sample, we'll resample several times from here
        bhyplen = len(tok(orig_os[0][bscoind]).input_ids) 
        inps = {
            'allhyps':[[orig_os[0][bscoind]]*tsamps[0]],
            'allscos':[orig_scos[0]]
        }
        for j in range(3, bhyplen):
            prompt, orig_os, orig_scos, nav, oav, obests, oavgs, nb, ob, stats = sampfrominp(datadf, tok, mod, stmtok, stmshp, i, 0, inps, [j, -1], rchoose, tsamps, temp)
            allvals.append({
                'inp':prompt,
                'hyps':orig_os[1],
                'scos':orig_scos[1],
                "stats":stats[1],
                "ver":"best",
                "prefix":j
            })
            # save progress every 10
            if j%10==0:
                tmp = pd.DataFrame(allvals)
                tmp.to_json("output/exhaustive/ex"+str(i)+".jsonl", orient="records", lines=True)

        wscoind = np.argmin(orig_scos[0])
        # take len of worst path in initial sample, we'll resample several times from here
        whyplen = len(tok(orig_os[0][wscoind]).input_ids) 
        inps = {
            'allhyps':[[orig_os[0][wscoind]]*tsamps[0]],
            'allscos':[orig_scos[0]]
        }
        for j in range(3, whyplen):
            prompt, orig_os, orig_scos, nav, oav, obests, oavgs, nb, ob, stats = sampfrominp(datadf, tok, mod, stmtok, stmshp, i, 0, inps, [j, -1], rchoose, tsamps, temp)
            allvals.append({
                'inp':prompt,
                'hyps':orig_os[1],
                'scos':orig_scos[1],
                "stats":stats[1],
                "ver":"worst",
                "prefix":j
            })
            # save progress every 10
            if j%10==0:
                tmp = pd.DataFrame(allvals)
                tmp.to_json("output/exhaustive/ex"+str(i)+".jsonl", orient="records", lines=True)

        tmp = pd.DataFrame(allvals)
        tmp.to_json("output/exhaustive/ex"+str(i)+".jsonl", orient="records", lines=True)

        #sampfrominp(datadf, tok, mod, stmtok, stmshp, i, 0, inps, [-1], rchoose, tsamps, temp)
        #tmp = pd.DataFrame(allvals)
        #tmp.to_json("output/prefpreds/verbprefpreds"+str(numind)+"_"+str(pflen[0])+".jsonl", lines=True, orient="records")
    return tmp

def inpsampall(datadf, tok, mod, stmtok, stmshp, inps, pflen, rchoose, tsamps, numind, temp):
    allvals = []
    for i in range(len(datadf)):
        prompt, orig_os, orig_scos, nav, oav, obests, oavgs, nb, ob, stats = sampfrominp(datadf, tok, mod, stmtok, stmshp, i, numind, inps, pflen, rchoose, tsamps, temp)
        allvals.append({
            'inp':prompt,
            'hyps':orig_os,
            'scos':orig_scos,
            'new_avg':nav,
            "old_avg":oav,
            "newmax":nb,
            "oldmax":ob,
            "obests":obests,
            "oavgs":oavgs,
            "stats":stats
        })
        tmp = pd.DataFrame(allvals)
        tmp.to_json("output/prefpreds/verbprefpreds"+str(numind)+"_"+str(pflen[0])+".jsonl", lines=True, orient="records")
    return tmp

def mean(l):
    return sum(l)/len(l)

def dset_randsamp(datadf, tok, mod, stmtok, stmshp, rchoose, tsamps, temp, resamp=False):
    allvals = []
    for i in range(len(datadf)):
        # get initial sample to go off of
        prompt, orig_os, orig_scos, nav, oav, obests, oavgs, nb, ob, stats = sampfrominp(datadf, tok, mod, stmtok, stmshp, i, 0, None, [2], [rchoose[0]], [tsamps[0]], temp)
        allvals.append({
            'inp':prompt,
            'hyps':orig_os[0],
            'scos':orig_scos[0],
            "stats":stats[0],
            "ver":"first",
            "pref":0
        })
        if resamp:
            # take a random hyp
            bscoind = random.randint(0, len(orig_scos[0])-1)
            # take len of best path in initial sample, we'll resample several times from here
            bhyplen = len(tok(orig_os[0][bscoind]).input_ids) 
            j = random.randint(3, bhyplen-1)
            inps = {
                'allhyps':[[orig_os[0][bscoind]]*tsamps[0]],
                'allscos':[orig_scos[0]]
            }
            # only do resample every 3 tokens
            
            prompt, orig_os, orig_scos, nav, oav, obests, oavgs, nb, ob, stats = sampfrominp(datadf, tok, mod, stmtok, stmshp, i, 0, inps, [j, -1], rchoose, tsamps, temp)
            allvals.append({
                'inp':prompt,
                'hyps':orig_os[1],
                'scos':orig_scos[1],
                "stats":stats[1],
                "ver":"best",
                "prefix":j
            })
        # save progress every 10 samples
        if i%20==0:
            tmp = pd.DataFrame(allvals)
            tmp.to_json("output/biggerdset2.jsonl", orient="records", lines=True)

        #sampfrominp(datadf, tok, mod, stmtok, stmshp, i, 0, inps, [-1], rchoose, tsamps, temp)
        #tmp = pd.DataFrame(allvals)
        #tmp.to_json("output/prefpreds/verbprefpreds"+str(numind)+"_"+str(pflen[0])+".jsonl", lines=True, orient="records")
    return tmp