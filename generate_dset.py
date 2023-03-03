from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.models.models import load_from_checkpoint as lfc
from src.utils.score_utils import metrics_mapping
from src.utils.generation_utils import load_model, all_tfr_decode
import matplotlib.pyplot as plt

import logging
import pandas as pd
import torch
import time
import math

def weightdecay(numtoks):
    return 5/numtoks

def onlytfr(numtoks):
    return 0

#allcands = pd.read_csv("tmp.csv", index_col=0)
def rer_met(rer, tgt, df):
    N=1
    fsort = df.sort_values(by=['ref', rer], ascending=[True, False]).groupby('ref', as_index=False).nth[:N]
    return fsort[tgt].mean()

    # get results in a df for a given configuration 
def test_config(mod, tok, data, config):
    allcands = all_tfr_decode(mod, tok, data, config)
    allcands['modsco'] = [float(f) for f in allcands['modsco']]
    #metrics_mapping("pqe", allcands)
    metrics_mapping("parent",allcands)
    return allcands

if __name__=="__main__":
    setting = "table2text"
    # get generation model with monkey patching based on setting
    mod, tok, dset, dec_pref = load_model(setting, True, "cuda:1", True)
    dset = list(dset)

    save_interv = 2000
    saves = int(math.ceil(len(dset)/save_interv))
    adfs = []
    for s in range(saves):
        allcands = test_config(mod, tok, dset[save_interv*s:(s+1)*save_interv], {
                "max_len":90,
                "device":'cuda:1',
                "beam_size":50,
                "dec_prefix":dec_pref,
                "tfr_interv":1000,
                "tfr_beams":13,
                "weightfunc":onlytfr
        })
        adfs.append(allcands)
        wholedf = pd.concat(adfs)
        wholedf.to_csv("dset_tmp.csv")
    wholedf.to_csv("dset_full.csv")

    