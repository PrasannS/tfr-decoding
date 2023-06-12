import numpy as np
import pandas as pd
from statistics import stdev, mean
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl")

import spacy

# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_sm")

def extract_scatter(datadf, use):
    uns = datadf['inp'].unique()
    allstats = {
        'mean':[],
        'stdev':[],
        'entropy':[],
        'skewness':[],
        'kurtoses':[],
        'top1s':[],
        'selected':[],
        'inpstrs':[],
        'improves':[],
        'initsco':[],
        'prefpercent':[],
        'prefix':[],
        'selstrs':[],
        'prompt':[],
        'stdevs':[],
        'means':[],
        'phraseprefs':[],
        'contprefs':[],
        "postags":[],
    }
    ind = 0
    for u in uns:
        print(ind)
        tmpdf = datadf[datadf['inp']==u]
        wentry = tmpdf[tmpdf['ver']=='first'].iloc[0]
        wtmp = tmpdf[tmpdf['ver']==use].copy()
        wtmp['prefix'] = wtmp['prefix'] - 1
        wind = bmatch(tmpdf, wentry['hyps'])
        wtoks = tokenizer(wentry['hyps'][wind]).input_ids
        
        # get all phrase, entity-based boundaries 
        # todo validate if this actually works
        allstats['phraseprefs'].extend([get_phrase_prefs(wentry['hyps'][wind])]*len(wtmp))
        allstats['contprefs'].extend([get_content_prefs(wentry['hyps'][wind])]*len(wtmp))
        
        wtmp['max_scos'] = wtmp['scos'].apply(max)
        allstats['means'].extend(wtmp['scos'].apply(mean))
        allstats['stdevs'].extend(wtmp['scos'].apply(stdev))
        
        for a in list(allstats.keys())[:7]:
            try:
                allstats[a].extend(wtmp.apply(lambda row: wentry['stats'][a][int(row['prefix'])][wind], axis=1))
            except:
                print(a, " failed")
                print("max prefix is ", max(wtmp['prefix']))
                print("stats len is ", len(wentry['stats'][a]))
        
        wtmp['istrs'] = wtmp.apply(lambda row: [tokenizer.decode(w) for w in wtoks], axis=1)
        postags = get_pos_tags(wentry['hyps'][wind])
        allstats['postags'].extend([postags]*len(wtmp))
                               
        allstats['inpstrs'].extend(wtmp['istrs'])
        allstats['improves'].extend(wtmp['max_scos'] - wentry['scos'][wind])
        allstats['initsco'].extend([wentry['scos'][wind]]*len(wtmp))
        allstats['prefpercent'].extend(wtmp['prefix']/len(wtmp['istrs'].iloc[0]))
        allstats['prefix'].extend(wtmp['prefix'])
        
        wtmp['sel'] = wtmp.apply(lambda row: np.argmax(row['scos']), axis=1)
        seltmp = wtmp.apply(lambda row: [tokenizer(row['hyps'][row['sel']]).input_ids], axis=1)
        seltmp = [[tokenizer.decode(w) for w in sel] for sel in seltmp]
        allstats['selstrs'].extend(seltmp)
        allstats['prompt'].extend(wtmp['inp'])
        ind+=1

    return pd.DataFrame(allstats)

def bmatch(tmpdf, istrs):
    rhyp = tmpdf.loc[tmpdf['prefix'].idxmax()]['hyps'][0]
    imatches = []
    for ist in istrs:
        i = 0
        while i<len(rhyp) and i<len(ist) and rhyp[i]==ist[i]:
            i = i+1
        imatches.append(i)
    result = int(np.argmax(imatches))
    #print(rhyp)
    #print(istrs[result])
    return result
        
    
def get_traintest(rdf, ratio):
    auns = rdf['prompt'].unique()
    cutind = int(len(auns)*ratio)+1
    tmp = rdf[rdf['prompt']==auns[cutind]].reset_index()
    cutind = int(tmp.iloc[0]['index'])
    return rdf.iloc[:cutind], rdf.iloc[cutind:]

def hastok(istr, start, end, tok):
    return int(tok in istr[start-2:end+1])

def aggregate_clusters(df, scocol, threshold):
    df = df.sort_values(by=['prompt', 'prefix']).reset_index(drop=True)
    aggregated_data = []
    
    for groupid, group in df.groupby('prompt'):
        startind = group.iloc[0]['prefix']
        endind = startind
        cluster_scores = [group.iloc[0][scocol]]
        
        for i in range(1, len(group)):
            row = group.iloc[i]
            score = row[scocol]
            index = row['prefix']

            cluster_mean = sum(cluster_scores) / len(cluster_scores)
            if abs(score - cluster_mean) <= threshold and (endind-startind)<15:
                endind = index
                cluster_scores.append(score)
            else:
                aggregated_data.append({
                    'scomean': cluster_mean,
                    'startind': startind,
                    'endind': endind,
                    'groupid': groupid, 
                    'initsco': group.iloc[0]['initsco'],
                    'inpstrs': group.iloc[0]['inpstrs'],
                    'phraseprefs': group.iloc[0]['phraseprefs'],
                    'contprefs': group.iloc[0]['contprefs'],
                })
                startind = index
                endind = index
                cluster_scores = [score]

        aggregated_data.append({
            'scomean': sum(cluster_scores) / len(cluster_scores),
            'startind': startind,
            'endind': endind,
            'groupid': groupid,
            'initsco': group.iloc[0]['initsco'],
            'inpstrs': group.iloc[0]['inpstrs'],
            'phraseprefs': group.iloc[0]['phraseprefs'],
            'contprefs': group.iloc[0]['contprefs'],
        })
    
    return pd.DataFrame(aggregated_data)


# CODE for phrase boundary stuff
def get_phrase_boundaries(doc):
    boundaries = []
    for token in doc:
        if token.dep_ in ["ROOT", "conj"]: #or token.pos_ in ["VERB", "ADJ", "NOUN", "PROPN"]:
            boundaries.append(token.i)
    return boundaries

# entity based boundaries
def get_content_boundaries(doc):
    boundaries = []
    for ent in doc.ents:
        boundaries.append(ent.end - 1)
    return boundaries

# extract to prefix strings so we can convert back to tokens for consistent indexing
def get_prefix_strings(text, boundaries, doc):
    prefixes = []
    for boundary in boundaries:
        prefixes.append(text[:doc[boundary].idx + len(doc[boundary].text)])
    return prefixes

def get_pref_inds(prefixes):
    inds = []
    for p in prefixes:
        # get index based on removing special tokens.
        inds.append(len(tokenizer(p).input_ids)-2)
    return inds

def get_phrase_prefs(text):
    doc = nlp(text)
    phrase_boundaries = get_phrase_boundaries(doc)
    return get_pref_inds(get_prefix_strings(text, phrase_boundaries, doc))

def get_content_prefs(text):
    doc = nlp(text)
    content_boundaries = get_content_boundaries(doc)
    return get_pref_inds(get_prefix_strings(text, content_boundaries, doc))

def get_pos_tags(text):
    doc = nlp(text)
    flan_tokens_pos_tags = []
    last_flan_token_index = -1

    for token in doc:
        # Create a substring up to the end of the current spaCy token
        substring = text[:token.idx + len(token.text)]

        # Tokenize the substring using the Flan tokenizer
        input_ids = tokenizer.encode(substring, add_special_tokens=False)

        # Calculate the number of new Flan tokens in this substring
        new_flan_token_count = len(input_ids) - last_flan_token_index - 1

        # Assign the POS tag of the current spaCy token to the new Flan tokens
        flan_tokens_pos_tags.extend([token.pos_] * new_flan_token_count)
        
        # Update the last Flan token index
        last_flan_token_index = len(input_ids) - 1

    # Assign remaining unassigned Flan tokens the POS tag of the next assigned Flan token
    for i in range(len(flan_tokens_pos_tags) - 1, -1, -1):
        if flan_tokens_pos_tags[i] is None:
            flan_tokens_pos_tags[i] = flan_tokens_pos_tags[i + 1]

    return flan_tokens_pos_tags