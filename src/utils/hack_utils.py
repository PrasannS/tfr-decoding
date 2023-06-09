import re
from src.utils.samp_utils import gen_row, get_reward_single
import numpy as np
from nltk.tokenize import word_tokenize
from collections import Counter
from collections import defaultdict

device="cuda:1"

THRESH = -1
def get_class(score):
    if score <= THRESH:
        return 0
    else:
        return 1

def competency_analysis(scos, hyps):
    # Put the scores into two classes
    class_scores = [get_class(sco) for sco in scos]

    # Tokenize the hyps
    tokens = [word_tokenize(hyp) for hyp in hyps]

    # Associate each token with its class
    tokens_and_classes = zip(tokens, class_scores)
    
    # Create a dictionary to hold counts for each word in each class
    word_counts = defaultdict(lambda: [0, 0])  # index 0 for class 0, index 1 for class 1

    # Fill the dictionary
    for tokens, class_score in tokens_and_classes:
        checked = set()
        for token in tokens:
            if token not in checked:
                word_counts[token][class_score] += 1
                checked.add(token)

    # Compute the probabilities
    probabilities = {word: (counts[1] / sum(counts)) for word, counts in word_counts.items()}
    return probabilities, word_counts

    
# given hyps, scos lists, find pairs with gaps, use them to try different modifications, to test steamshp hypotheses
def modify_hyps(allscos, allhyps, allinps):
    result_dicts = []
    
    # go through the whole thing
    for index in range(len(allscos)):
        # get relevant lists
        scos, hyps = allscos[index], allhyps[index]
        tmp_rdicts = []
        # look for a pair
        for i in range(len(scos)):
            for j in range(0, i+1):
                # look for things with gap of .2
                if abs(scos[i] - scos[j]) >= 0.2 and len(tmp_rdicts)==0:
                    lower_hyp, higher_hyp = (hyps[i], hyps[j]) if scos[i] < scos[j] else (hyps[j], hyps[i])
                    lower_sco, higher_sco = (scos[i], scos[j]) if scos[i] < scos[j] else (scos[j], scos[i])

                    lower_hyp_sentences = re.split('(?<=[.!?]) +', lower_hyp)
                    higher_hyp_sentences = re.split('(?<=[.!?]) +', higher_hyp)

                    if len(lower_hyp_sentences) >= 2:
                        rdict = {}
                        rdict['lsent'] = lower_hyp
                        rdict['hsent'] = higher_hyp
                        rdict['lsco'] = lower_sco
                        rdict['hsco'] = higher_sco
                        first_sentence_higher = higher_hyp_sentences[0]
                        last_sentence_higher = higher_hyp_sentences[-1]
                        
                        rdict['first_add'] = ' '.join([first_sentence_higher]+lower_hyp_sentences)
                        rdict['last_add'] = ' '.join(lower_hyp_sentences+[last_sentence_higher])
                        fsent_lower = lower_hyp_sentences[0]
                        lower_hyp_sentences[0] = first_sentence_higher
                        rdict['first_rep'] = ' '.join(lower_hyp_sentences)
                        lower_hyp_sentences[0] = fsent_lower
                        lower_hyp_sentences[-1] = last_sentence_higher
                        rdict['last_rep'] = ' '.join(lower_hyp_sentences)
                        rdict['dup_lower'] = lower_hyp+" "+lower_hyp
                        tmp_rdicts.append(rdict)
        if len(tmp_rdicts)>0:
            result_dicts.append(tmp_rdicts[0])
            result_dicts[-1]['inp'] = allinps[index]
    return result_dicts


def getscore(q, a, rm, tok, bigmodel=True):
    if bigmodel: 
        input_text = "<|prompter|>"+q+"<|endoftext|><|assistant|>"+a+"<|endoftext|>"
        inputs = tok(input_text, return_tensors="pt").to(device)
    else:
        inputs = tok(q, a, return_tensors='pt').to(device)
    score = rm(**inputs).logits[0].cpu().detach()
    return score

# score output of above method
def score_mhyp(mhyp, steamtok, steamshp, isoa=False):
    resdict = {}
    for k in mhyp.keys():
        resdict[k] = mhyp[k]
        if '_' in k and "sco" not in k:
            #print(k)
            if isoa:
                resdict[k+"_score"] = float(getscore(mhyp['inp'], mhyp[k], steamshp, steamtok))
            else:
                resdict[k+"_score"] = float(get_reward_single({"context": mhyp['inp'], "hyp":mhyp[k]}, steamtok, steamshp))
    return resdict

def flesch_kincaid(text):
    num_sentences = len(re.findall(r'[.!?]+', text))
    num_words = len(re.findall(r'\w+', text))
    num_syllables = sum([len(re.findall(r'[aeiouy]+', word, re.IGNORECASE)) for word in re.findall(r'\w+', text)])

    if num_sentences == 0 or num_words == 0:
        
        num_sentences=1
        num_words = max(num_words, 1)

    score = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)
    return score