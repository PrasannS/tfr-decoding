from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from datasets import load_dataset
import pandas as pd
import numpy as np
from src.tfr_decoding.prefix_sample import sample as naivesample# new sampling method
from src.tfr_decoding.finepref_sample import sample as fpsample
from src.tfr_decoding.adapt_pfsample import sample as apsample
from src.tfr_decoding.naive_plus_sample import sample as ensample
from src.utils.samp_utils import inpsampall, dset_randsamp   
from src.tfr_decoding.shp_modeling import T5BinaryClassifier

import sys
sys.setrecursionlimit(1500) # this number can be any limit


class PrefixSampler():
    def __init__(self, genmodel, num_hyps=1, temp=.9):
        self.mod = genmodel
        self.temp = temp
        self.num_hyps = num_hyps
    
    # make prompt for eli5 generation
    def construct_gen_prompt(self, source):
        template = \
    """
    Give a lengthy, detailed response. 
    Question: """
        inp = template+source+"\n Detailed Response:"
        
        
        return inp
    
    # generate output for an input row
    def gen_row(self, source):
        input_text = self.construct_gen_prompt(source)
        
        #print(input_text)
        input_ids = self.mod.tok(input_text, return_tensors="pt").input_ids.to(self.mod.device)
        # only do sampling
        outputs = self.mod.generate(input_ids, min_new_tokens=20, max_new_tokens=300, do_sample=True, top_p=.95, temperature=self.temp, num_return_sequences=self.num_hyps
                                    , return_dict_in_generate=True, output_scores=True)
        outs = [self.mod.tok.decode(o, skip_special_tokens=True) for o in outputs.sequences]

        return source, outs, outputs.scores

    # score a single example (I don't think there's enough space to batch this?)
    def get_reward_single(self, inpdict):
        template = "POST: {context:s} \n\nRESPONSE A:{hyp:s} \n\nRESPONSE B: .\n\n Which response is better? RESPONSE "
        inp = template.format(context=inpdict['context'], hyp=inpdict['hyp'])
        x = self.mod.downtok([inp], return_tensors='pt').input_ids.to(self.mod.downmetric.device)
        outputs = self.mod.downmetric.generate(x, return_dict_in_generate=True, output_scores=True, max_new_tokens=1)
        return torch.exp(outputs.scores[0][:, 71]) / torch.exp(outputs.scores[0][:,:]).sum(axis=1).item() # index 71 corresponds to the token for 'A'
    
    # keep sampling until we get something good enough or hit max rounds
    def adaptive_baseline(self, source, thresh=0.85, rounds=4):
        scores = []
        allouts = []
        for i in range(rounds):
            _, outs, _ = self.gen_row(source)
            score = self.get_reward_single({'context':source, 'hyp':outs[0]})
            scores.append(float(score))
            allouts.append(outs[0])
            tot_toks = sum([len(self.mod.tok(o).input_ids) for o in allouts])
            # we don't need to sample anymore
            if score>thresh:
                return scores, allouts, tot_toks
        
        
        return scores, allouts, tot_toks
    
    # adaptive sampling, but this time we re-sample from promising prefixes with each new round instead of 
    # random new stuff
    def adapt_pfsample(self, source, thresh, rounds, decay=0.9, rec_n=3):
        scores = []
        allouts = []
        self.mod.sample = apsample.__get__(self.mod)
        self.mod.over_inpids=None
        self.mod.decoded_toks = 0
        self.mod.decay = decay
        self.mod.rec_n = rec_n
        self.mod.source_str = source
        for i in range(rounds):
            # generate, get score, decoding algorithm automatically handles new 
            # resample point 
            _, outs, _ = self.gen_row(source)
            score = self.get_reward_single({'context':source, 'hyp':outs[0]})
            print(score, " ", outs[0])
            scores.append(float(score))
            allouts.append(outs[0])
            tot_toks = self.mod.decoded_toks
            # we don't need to sample anymore
            if score>thresh:
                self.mod.decoded_toks = 0
                return scores, allouts, tot_toks
        
        self.mod.decoded_toks = 0
        return scores, allouts, tot_toks
    
    # do prefix sampling following algorithm defined from before
    def do_prefix_sample(self, source, max_resamps=6, checks=[15]):
        self.mod.sample = naivesample.__get__(self.mod)
        self.mod.source_str = source
        # once we hit max_resamps, then just decode to end with what we have (TODO might need a better baseline?)
        self.mod.max_resamps = max_resamps
        # checkpoint # of tokens 
        self.mod.checklist = checks
        _, outs, _ = self.gen_row(source)
        score = self.get_reward_single({'context':source, 'hyp':outs[0]})
        dectoks = self.mod.decoded_toks
        self.mod.decoded_toks = 0
        return outs[0], float(score), dectoks
    
    # do prefix sampling following algorithm defined from before
    def do_enhanced_sample(self, source, max_resamps=6, checks=[15], rec_n=3, cont_checks=3, c_thresh=0.75, t_thresh=0.85, asampmax=3):
        self.mod.sample = ensample.__get__(self.mod)
        self.mod.source_str = source
        # once we hit max_resamps, then just decode to end with what we have (TODO might need a better baseline?)
        self.mod.max_resamps = max_resamps
        # checkpoint # of tokens 
        self.mod.checklist = checks
        # how frequently to get checkpoints
        self.mod.rec_n = rec_n
        self.mod.cont_checks = cont_checks
        _, outs, _ = self.gen_row(source)
        score = self.get_reward_single({'context':source, 'hyp':outs[0]})
        dectoks = self.mod.decoded_toks
        self.mod.decoded_toks = 0
        self.mod.c_thresh = c_thresh # conf thresh
        self.mod.t_thresh = t_thresh # for adaptive fallback
        self.mod.asampmax = asampmax # how much to go for adaptive sampling
        return outs[0], float(score), dectoks, self.mod.allstrs, self.mod.allscos
    
    def do_fine_sample(self, source, max_resamps, rec_n = 3, check_n = 3, cont_len = 5):

        self.mod.sample = fpsample.__get__(self.mod)
        self.mod.source_str = source
        # once we hit max_resamps, then just decode to end with what we have (TODO might need a better baseline?)
        self.mod.max_resamps = max_resamps
        print("setting values")
        # checkpoint # of tokens 
        self.mod.rec_n = rec_n
        self.mod.check_n = check_n
        self.mod.cont_len = cont_len
        _, outs, _ = self.gen_row(source)
        score = self.get_reward_single({'context':source, 'hyp':outs[0]})
        dectoks = self.mod.decoded_toks
        self.mod.decoded_toks = 0
        return outs[0], float(score), dectoks
    
SAVEINT = 50
def test_baseline(inplist, pfsampler, thresh, rounds, fname):
    ascos = []
    abudgets = []
    allouts = []
    for inp in inplist:
        try:
            scos, outs, tot_toks = pfsampler.adaptive_baseline(inp, thresh, rounds)
            best_ind = int(np.argmax(scos))
            ascos.append(scos[best_ind])
            allouts.append(outs[best_ind])
            abudgets.append(tot_toks)
            
            if ((len(abudgets)+1)%SAVEINT)==0 and fname is not None:
                tmp = pd.DataFrame({"scos":ascos, "budgets":abudgets, "outs":allouts})
                tmp.to_json(fname, orient="records", lines=True)
        except:
            print("strange issue")
            ascos.append(0)
            allouts.append("")
            abudgets.append(-1)
        print(len(ascos))
            
    return pd.DataFrame({"scos":ascos, "budgets":abudgets, "outs":allouts})

def test_apsample(inplist, pfsampler, thresh, rounds, decay, rec_n, fname):
    ascos = []
    abudgets = []
    allouts = []
    for inp in inplist:
        try:
            # ideally should function as lower budget version of adaptive sample
            scos, outs, tot_toks = pfsampler.adapt_pfsample(inp, thresh, rounds, decay, rec_n)
            best_ind = int(np.argmax(scos))
            ascos.append(scos[best_ind])
            allouts.append(outs[best_ind])
            abudgets.append(tot_toks)
            
            if ((len(abudgets)+1)%SAVEINT)==0 and fname is not None:
                tmp = pd.DataFrame({"scos":ascos, "budgets":abudgets, "outs":allouts})
                tmp.to_json(fname, orient="records", lines=True)
        except:
            ascos.append(0)
            allouts.append("")
            abudgets.append(-1)
    return pd.DataFrame({"scos":ascos, "budgets":abudgets, "outs":allouts})

def test_pfsample(inplist, pfsampler, max_resamps, checks, fname):
    ascos = []
    abudgets = []
    allouts = []
    for inp in inplist:
        try:
            out, score, dectoks = pfsampler.do_prefix_sample(inp, max_resamps, checks)
            ascos.append(score)
            allouts.append(out)
            abudgets.append(dectoks)
            if ((len(abudgets)+1)%SAVEINT)==0 and fname is not None:
                tmp = pd.DataFrame({"scos":ascos, "budgets":abudgets, "outs":allouts})
                tmp.to_json(fname, orient="records", lines=True)
        except:
            ascos.append(0)
            allouts.append("")
            abudgets.append(-1)
    return pd.DataFrame({"scos":ascos, "budgets":abudgets, "outs":allouts})

def test_enhancedsample(inplist, pfsampler, max_resamps, checks, rec_n, cont_checks, c_thresh, t_thresh, asampmax, fname):
    ascos = []
    abudgets = []
    allouts = []
    a_ascos = []
    a_astrs = []
    for inp in inplist:
        try:
            out, score, dectoks, ast, asc = pfsampler.do_enhanced_sample(inp, max_resamps, checks, rec_n, cont_checks, c_thresh, t_thresh, asampmax)
            a_ascos.append(asc)
            a_astrs.append(ast)
            print(dectoks)
            print(score)
            ascos.append(score)
            allouts.append(out)
            abudgets.append(dectoks)
            if ((len(abudgets)+1)%SAVEINT)==0 and fname is not None:
                tmp = pd.DataFrame({"scos":ascos, "budgets":abudgets, "outs":allouts, "allscos":a_ascos, "allstrs":a_astrs})
                tmp.to_csv(fname)
        except:
            print("something wrong")
            ascos.append(0)
            allouts.append("")
            abudgets.append(-1)
            torch.cuda.empty_cache()
    return pd.DataFrame({"scos":ascos, "budgets":abudgets, "outs":allouts, "allscos":a_ascos, "allstrs":a_astrs})

def test_finesample(inplist, pfsampler, max_resamps, rec_n = 3, check_n = 3, cont_len = 5, fname=None):
    ascos = []
    abudgets = []
    allouts = []
    for inp in inplist:
        try:
            out, score, dectoks = pfsampler.do_fine_sample(inp, max_resamps, rec_n, check_n, cont_len)
            ascos.append(score)
            allouts.append(out)
            abudgets.append(dectoks)
            print(score)
            if ((len(abudgets)+1)%SAVEINT)==0 and fname is not None:
                tmp = pd.DataFrame({"scos":ascos, "budgets":abudgets, "outs":allouts})
                tmp.to_json(fname, orient="records", lines=True)
        except:
            ascos.append(0)
            allouts.append("")
            abudgets.append(-1)
    return pd.DataFrame({"scos":ascos, "budgets":abudgets, "outs":allouts})

if __name__=="__main__":
    # first load relevant models
    device = 'cuda:0' # if you have a GPU


    # get generation model
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
    qpref = T5BinaryClassifier.load_from_checkpoint("./lightning_logs/version_4/checkpoints/epoch=2-step=11896.ckpt")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map="auto")#.to(device)
    
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
    elidf = elidf.drop_duplicates(subset="history").iloc[15000:15100]

    pfsampler = PrefixSampler(model)
    inplist = list(elidf['history'])
    
    finesample_df = test_finesample(inplist, pfsampler, 9, 3, 6, 5)

    #adaptbase_df = test_baseline(inplist, pfsampler, .85, 1)
    #pfsample_df = test_pfsample(inplist, pfsampler, 18, [5, 25])

    #adaptbase_df.to_json("output/pfsample/abase.jsonl", lines=True, orient='records')
    #pfsample_df.to_json("output/pfsample/pfsample2.jsonl", lines=True, orient='records')
    finesample_df.to_json("output/pfsample/finesamp.jsonl", lines=True, orient='records')
    