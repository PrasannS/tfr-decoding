import torch
import numpy as np
import pandas as pd
from sigfig import round as rnd
import random


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
def get_reward_single(inpdict, steamtok, steamshp):
    template = "POST: {context:s} \n\nRESPONSE A:{hyp:s} \n\nRESPONSE B: .\n\n Which response is better? RESPONSE "
    inp = template.format(context=inpdict['context'], hyp=inpdict['hyp'])
    x = steamtok([inp], return_tensors='pt').input_ids.to(steamshp.device)
    outputs = steamshp.generate(x, return_dict_in_generate=True, output_scores=True, max_new_tokens=1)
    return torch.exp(outputs.scores[0][:, 71]) / torch.exp(outputs.scores[0][:,:]).sum(axis=1).item() # index 71 corresponds to the token for 'A'

# generate output for an input row
def gen_row(rw, tok, mod, method="greedy", num_hyps=10, temp=.9, ismult=False):
    
    if ismult:
        input_text = []
        for r in range(len(rw)):
            input_text.append(construct_prompt(rw.iloc[r]))
    else:
        input_text = construct_prompt(rw)
    
    #handles batching, TODO check if it works
    input_ids = tok(input_text, return_tensors="pt", padding=True, truncation=True).input_ids.to(mod.device)
    if method=="greedy":
        outputs = mod.generate(input_ids, min_new_tokens=20, max_new_tokens=200)
        outs = [tok.decode(outputs[0], skip_special_tokens=True)]
    elif method=="sample": 
        outputs = mod.generate(input_ids, min_new_tokens=20, max_new_tokens=300, do_sample=True, top_p=.95, temperature=temp, num_return_sequences=num_hyps, return_dict_in_generate=True, output_scores=True)
        outs = [tok.decode(o, skip_special_tokens=True) for o in outputs.sequences]
    elif method=="beam":
        outputs = mod.generate(input_ids, min_new_tokens=20, max_new_tokens=200, num_beams=num_hyps, num_return_sequences=num_hyps)
        outs = [tok.decode(o, skip_special_tokens=True) for o in outputs]
    lens = [len(tok(o).input_ids) for o in outs]
    if ismult:
        fouts = []
        fstats = []
        flens = []
        # divide result by input
        for i in range(int(len(outs)/num_hyps)):
            fouts.append(outs[i*num_hyps:(i+1)*num_hyps])
            #fstats.append(outputs.scores[i*num_hyps:(i+1)*num_hyps])
            flens.append(lens[i*num_hyps:(i+1)*num_hyps])
        return [list(rw['history'])], fouts, None, flens
    return [rw['history']], [outs], None, lens

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


def gen_rec_samp(rw, tok, mod, stmtok, stmshp, pfs, keepR, tsamps, startinp=None, srow=-1, temp=.9, isbatch=False):
    mod.man_pref=None
    # generate with initial sample
    if isbatch:
        rsize = len(rw)
    else:
        rsize = 1
        startinp = [startinp]
    results = [{'ihyps':[], 'iscos':[], 'scodfs':[], 'lens':[]} for i in range(rsize)]
    for i in range(len(keepR)):
        # use previous decoding
        if startinp and None not in startinp and i==0:
            # take inputs separately for everything
            for j in range(rsize):
                results[j]['ihyps'].append(startinp[j]["allhyps"][srow])
                results[j]['iscos'].append(startinp[j]['allscos'][srow])
                #results[j]['scodfs'].append(None)
                results[j]['lens'].append(None)
        else:
            # should work for single and batched?
            inp, outs, scoredf, lens = gen_row(rw, tok, mod, "sample", tsamps[i], temp, isbatch)
            for j in range(rsize):
                
                # generate scores to re-rank, only use best options for next step
                shp_scores = [float(get_reward_single({"context": inp[j], "hyp":o}, stmtok, stmshp)) for o in outs[j]]
                #for k in scoredf[j].keys():
                    # convert to floats for easier storage
                #    scoredf[j][k] = [[rnd(c.item(), 4) for c in row] for row in scoredf[k]]
                results[j]['ihyps'].append(outs[j])
                results[j]['iscos'].append(shp_scores)
                #results[j]['scodfs'].append(scoredf[j])
                results[j]['lens'].append(lens[j])
        forceprefs = []
        for j in range(rsize):
            # sort by shp scores
            bestopts = list(np.argsort(results[j]['iscos'][-1]))
            otmps = results[j]['ihyps'][-1]
            bestopts.reverse()
            bestopts = bestopts[:keepR[i]]
            nouts = ["<pad>"+otmps[bo].strip() for bo in bestopts]
            
            # option for manual token
            if pfs[i]>1:
                prefcut=pfs[i]
            else:
                # calculate prefix lens for prefcut calculation
                spls = [nout.split() for nout in nouts]
                #spls = [sp[:int(len(sp)*pfs[i])] for sp in spls]
                splens = [len(l) for l in spls]
                print(splens)
                prefcut = int(int(sum(splens)/len(splens))*pfs[i])
            print("prefcut ", prefcut)
            # TODO setup for multiple inputs
            forced = tok(nouts, return_tensors="pt", padding=True, truncation=True).input_ids.to(mod.device)[:, :-1]
            forced = forced[:, :prefcut]
            if i==len(tsamps)-1:
                tlen = tsamps[i]
            else:
                tlen = tsamps[i+1]
            # NOTE we'll constrain so all sequences of batch use same manual prefix
            forced = forced.repeat(int(tlen/keepR[i]), 1)
            print("PREFIX: ")
            print(tok.batch_decode(forced))
            forceprefs.append(forced)
        if pfs[i]>0:
            # throw in all prefixes at one go? TODO check how it's actually done
            mod.man_pref = torch.cat(forceprefs, dim=0)
    
    return results

def sampfrominp(datadf, tok, mod, stmtok, stmshp, ind, row, inps, pflen, rchoose, tsamps, temp, batchsize=1):
    if batchsize>1:
        datatmp = datadf.iloc[ind:ind+batchsize]
    else:
        datatmp = datadf.iloc[ind]
    if inps:
        # throw in multiple when necessary
        results = gen_rec_samp(datatmp, tok, mod, stmtok, stmshp, pflen, rchoose, tsamps, inps, row, temp, batchsize>1)
    else:
        results = gen_rec_samp(datatmp, tok, mod, stmtok, stmshp, pflen, rchoose, tsamps, None, row, temp, batchsize>1)
    for r in range(len(results)):
        results[r]['newmax'] = max(results[r][-1]['iscos'])
        results[r]['oldmax'] = max(results[r][0]['iscos'])
        results[r]['new_avg'] = mean(results[r][-1]['iscos'])
        results[r]['old_avg'] = mean(results[r][0]['iscos'])
        results[r]['inp'] = datatmp.iloc[r]['history']
        print("newbest ", results[r]['newmax'], "; oldbest ", results[r]['oldmax'])
        print("newavg ", results[r]['new_avg'], "; oldavg", results[r]['old_avg'])
    
    return results

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
            #"stats":stats
        })
        tmp = pd.DataFrame(allvals)
        tmp.to_json("output/prefpreds/verbprefpreds"+str(numind)+"_"+str(pflen[0])+".jsonl", lines=True, orient="records")
    return tmp

def mean(l):
    return sum(l)/len(l)

def dset_randsamp(datadf, tok, mod, stmtok, stmshp, rchoose, tsamps, temp, batchsize=1):
    allvals = []
    for ind in range(int(len(datadf)/batchsize)):
        
        i = ind*batchsize
        # get initial sample to go off of
        firstresult = sampfrominp(datadf, tok, mod, stmtok, stmshp, i, 0, None, [2], [rchoose[0]], [tsamps[0]], temp, batchsize)
        
        bscoinds = []
        inps = []
        bhyplen = 10000
        for b in range(batchsize):
            
            allvals.append({
                'inp':firstresult[b]['inp'],
                'hyps':firstresult[b]['ihyps'][0],
                'scos':firstresult[b]['iscos'][0],
                #"stats":firstresult[b]['scodfs'][0],
                'lens':firstresult[b]['lens'][0], # use for prefix calculation
                "ver":"first",
                "pref":0
            })
            bhyplen = min(bhyplen, min(allvals[-1]['lens']))
            bscoinds.append(random.randint(0, len(tsamps[0])-1))
            bind = bscoinds[-1]
            inps.append({
                'allhyps':[firstresult[b]['ihyps'][0][bind]]*len(tsamps[0]),
                'allscos':[firstresult[b]['ihyps'][0][bind]]*len(tsamps[0])
            })
        
        # take a random hyp
        #bscoind = 
        # take len of best path in initial sample, we'll resample several times from here
        #bhyplen = len(tok(orig_os[0][bscoind]).input_ids) 
        #inps = 
        # just take 1 per thing
        try:
            # sample from a random prefix
            j = random.randint(3, bhyplen-1)
            prefresult = sampfrominp(datadf, tok, mod, stmtok, stmshp, i, 0, inps, [j, 1], rchoose, tsamps, temp)
            allvals.append({
                'inp':firstresult[b]['inp'],
                'hyps':firstresult[b]['ihyps'][1],
                'scos':firstresult[b]['iscos'][1],
                #"stats":firstresult[b]['scodfs'][1],
                'lens':firstresult[b]['lens'][1], # use for prefix calculation
                "ver":"rand",
                "pref":j
            })
        except:
            print("ran into some kind of error")
        # save progress every 10 samples
        if ind%10==0:
            tmp = pd.DataFrame(allvals)
            tmp.to_json("output/bigdset.jsonl", orient="records", lines=True)

        tmp = pd.DataFrame(allvals)
        tmp.to_json("output/bigdset.jsonl", orient="records", lines=True)
    return tmp