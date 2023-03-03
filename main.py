from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.models.models import load_from_checkpoint as lfc
from src.utils.score_utils import metrics_mapping
from src.utils.generation_utils import load_model, all_tfr_decode

import logging
import pandas as pd
import torch
import time

if __name__=="__main__":
    setting = "table2text"
    # get generation model with monkey patching based on setting
    mod, tok, dset, dec_pref = load_model(setting, "cuda:1")
    dset = list(dset)

    mod.man_pref = torch.tensor([0, 3])
    allcands = all_tfr_decode(mod, tok, dset[:2], {
        "max_len":90,
        "device":'cuda:1',
        "beam_size":12,
        "dec_prefix":dec_pref,
        "msco_ratio":1,
        "tfr_interv":5,
        "weightfunc":None,
        "tfr_beams":6
    })

    metrics_mapping("pqe", allcands)
    metrics_mapping("parent", allcands)