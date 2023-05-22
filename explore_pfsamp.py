from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from datasets import load_dataset
import pandas as pd
import numpy as np
from src.tfr_decoding.finepref_sample import sample # new sampling method
from src.utils.samp_utils import inpsampall, dset_randsamp   
from src.tfr_decoding.shp_modeling import T5BinaryClassifier
from prefix_sampling import PrefixSampler, test_baseline, test_pfsample, test_finesample, test_apsample, test_enhancedsample

# first load relevant models
device = 'cuda:0' # if you have a GPU
pfmod_path = "./lightning_logs/bestmodel/checkpoints/epoch=2-step=23950.ckpt"

# get generation model
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")

qpref = T5BinaryClassifier.load_from_checkpoint(pfmod_path)
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map="auto")#.to(device)
#model.sample = sample.__get__(model)
model.tokenizer = tokenizer
model.tok = tokenizer
pfname = 'stanfordnlp/SteamSHP-flan-t5-large'
max_len = 512
learning_rate = 3e-5
preftok = T5Tokenizer.from_pretrained(pfname)

model.qualitypref = qpref
#model.stdpref = T5BinaryClassifier.load_from_checkpoint("./lightning_logs/version_11/checkpoints/epoch=3-step=3359.ckpt")
# set up relevant mdoels
model.downmetric = T5ForConditionalGeneration.from_pretrained('stanfordnlp/SteamSHP-flan-t5-xl').to(device)
model.downtok = T5Tokenizer.from_pretrained('stanfordnlp/SteamSHP-flan-t5-xl')
model.eval()

# load up our dataset
elidf = pd.read_json("output/elidataset.jsonl", orient="records", lines="true")
# start from 15000 for latest exploration
elidf = elidf.drop_duplicates(subset="history").iloc[15000:15600]

pfsampler = PrefixSampler(model)
inplist = list(elidf['history'])

with torch.no_grad():
    # try out finesample
    #finesample_df = test_finesample(inplist, pfsampler, 9, 3, 6, 5)
    #finesample_df.to_json("output/hparam_explore/fsamp9_3_6_5.jsonl", orient="records", lines=True)
    
    #finesample_df = test_finesample(inplist, pfsampler, 18, 3, 3, 5)
    #finesample_df.to_json("output/hparam_explore/fsamp18_3_3_5.jsonl", orient="records", lines=True)
    
    #fname = "output/hparam_explore2/base1.jsonl"
    #finesample_df = test_baseline(inplist, pfsampler, .85, 1, fname)
    #finesample_df.to_json(fname, orient="records", lines=True)
    
    with torch.no_grad():
        fname = "output/hparam_explore2/enhsamp7_20.jsonl"
        finesample_df = test_enhancedsample(inplist, pfsampler, 18, [7, 20], 3, 3, fname)
        finesample_df.to_json(fname, orient="records", lines=True)
        
        fname = "output/hparam_explore2/enhsamp7_20_30.jsonl"
        finesample_df = test_enhancedsample(inplist, pfsampler, 18, [7, 20, 30], 3, 3, fname)
        finesample_df.to_json(fname, orient="records", lines=True)
    
    """
    
    fname = "output/hparam_explore2/base2.jsonl"
    finesample_df = test_baseline(inplist, pfsampler, .85, 2, fname)
    finesample_df.to_json(fname, orient="records", lines=True)
    
    fname = "output/hparam_explore2/base3.jsonl"
    finesample_df = test_baseline(inplist, pfsampler, .85, 3, fname)
    finesample_df.to_json(fname, orient="records", lines=True)
    
    fname = "output/hparam_explore2/base4.jsonl"
    finesample_df = test_baseline(inplist, pfsampler, .85, 4, fname)
    finesample_df.to_json(fname, orient="records", lines=True)
    
    fname = "output/hparam_explore2/base4v2.jsonl"
    finesample_df = test_baseline(inplist, pfsampler, .85, 4, fname)
    finesample_df.to_json(fname, orient="records", lines=True)
    
    fname = "output/hparam_explore2/pfsamp_5_20.jsonl"
    finesample_df = test_pfsample(inplist, pfsampler, 18, [5, 20], fname)
    finesample_df.to_json(fname, orient="records", lines=True)
    
    fname = "output/hparam_explore2/pfsamp_10_20.jsonl"
    finesample_df = test_pfsample(inplist, pfsampler, 18, [10, 20], fname)
    finesample_df.to_json(fname, orient="records", lines=True)
    
    fname = "output/hparam_explore2/pfsamp_10_20v2.jsonl"
    finesample_df = test_pfsample(inplist, pfsampler, 18, [10, 20], fname)
    finesample_df.to_json(fname, orient="records", lines=True)
    
    fname = "output/hparam_explore2/pfsamp_10_20_30.jsonl"
    finesample_df = test_pfsample(inplist, pfsampler, 18, [10, 20, 30], fname)
    finesample_df.to_json(fname, orient="records", lines=True)
    
    fname = "output/hparam_explore2/apsamp3_1_2.jsonl"
    finesample_df = test_apsample(inplist, pfsampler, 0.85, 3, 1, 3, fname)
    finesample_df.to_json(fname, orient="records", lines=True)
    
    fname = "output/hparam_explore2/apsamp3_1_2.jsonl"
    finesample_df = test_apsample(inplist, pfsampler, 0.85, 5, 1, 3, fname)
    finesample_df.to_json(fname, orient="records", lines=True)
    
    fname = "output/hparam_explore2/apsamp3_1_5.jsonl"
    finesample_df = test_apsample(inplist, pfsampler, 0.85, 3, 1, 5, fname)
    finesample_df.to_json(fname, orient="records", lines=True)
    
    fname = "output/hparam_explore2/pfsamp_5_20_30.jsonl"
    finesample_df = test_pfsample(inplist, pfsampler, 18, [5, 20, 30], fname)
    finesample_df.to_json(fname, orient="records", lines=True)
    
    fname = "output/hparam_explore2/fsamp9_3_3_5.jsonl"
    finesample_df = test_finesample(inplist, pfsampler, 9, 3, 3, 5, fname)
    finesample_df.to_json(fname, orient="records", lines=True)
    """
    
    
    