from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from datasets import load_dataset
import pandas as pd
import numpy as np
from src.tfr_decoding.custom_bs import beam_search
from src.tfr_decoding.recurse_samp import sample
from src.utils.samp_utils import inpsampall, dset_randsamp   

device = 'cuda:0' # if you have a GPU

elidf = pd.read_json("output/elidataset.jsonl", orient="records", lines="true")
# 16000 for P1
# 25000 for P2
# 31500 for P3
# 50000 for P4
elidf = elidf.drop_duplicates(subset="history").iloc[50000:]


# get shp model
steamtok = T5Tokenizer.from_pretrained('stanfordnlp/SteamSHP-flan-t5-xl')
steamshp = T5ForConditionalGeneration.from_pretrained('stanfordnlp/SteamSHP-flan-t5-xl').to(device)

# get generation model
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map="auto")#.to(device)
model.beam_search = beam_search.__get__(model)
model.sample = sample.__get__(model)
model.eval()


#fulleli5 = load_dataset("eli5")
#train = fulleli5['train_eli5']
#questions = []
#for i in range(len(train)):
#    f = train[i]
#    if f['selftext'] and len(f['selftext'])>10:
#        questions.append(f['title']+f['selftext'])

#elidf = pd.DataFrame({'history':questions})        
#eli5 = load_dataset("stanfordnlp/shp", data_dir="explainlikeimfive")
#eliorig = pd.DataFrame(eli5['train'])
#elidf = pd.read_json("output/elidataset.jsonl", orient="records", lines="true")
#elidf = elidf.drop_duplicates(subset="history").iloc[16000:]

#inpsall = pd.read_json("baselines1.jsonl", lines=True, orient='records')
#inpsall = pd.concat([inpsall, pd.read_json("baselines2.jsonl", lines=True, orient='records')])


pflen = [.7]
rchoose = [2]
tsamps = [3]

#sampfrominp(3, 0, inpsall, pflen, rchoose, tsamps)
#isall = inpsampall(elidf.iloc[:100], tokenizer, model, steamtok, steamshp, None, pflen, rchoose, tsamps, 0, 0.9)
#isall = inpsampall(elidf.iloc[:100], tokenizer, model, steamtok, steamshp, None, [.3, -1], rchoose, tsamps, 0, 0.9)

exsamp = dset_randsamp(elidf, tokenizer, model, steamtok, steamshp, rchoose, tsamps, 0.9)
#isall = inpsampall(inpsall.iloc[:50], pflen, rchoose, tsamps, 1, .9)
#isall = inpsampall(inpsall.iloc[:50], pflen, rchoose, tsamps, 2, .9)
#isall = inpsampall(inpsall.iloc[:50], pflen, rchoose, tsamps, 3, .9)
#isall = inpsampall(inpsall.iloc[:50], pflen, rchoose, tsamps, 4, .9)

