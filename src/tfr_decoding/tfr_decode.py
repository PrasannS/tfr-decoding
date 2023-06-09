import torch
from src.tfr_decoding.decode_node import DecodeNode
from src.tfr_decoding.util import run_inference_step
import math
import numpy as np

# somehow needs to encapsulate TFR decoding code

# return top args.beam_width possibilities given inputs
def decode_step(node: DecodeNode, gen_model, src_inp_ids, tokenizer, args):
    # get previous decodings based on stuff stored in node
    dec_inp_ids = torch.tensor([n.token_idx for n in node.prev]+[node.token_idx])
    output_prob, _, _ = run_inference_step(
            gen_model, src_inp_ids, decoder_input_ids=dec_inp_ids, device=src_inp_ids.device, output_dec_hid=False)
    # get the top k options by token index
    values, indices = torch.topk(output_prob, k=args["beam_size"])
    values = values[0].tolist()
    indices = indices[0].tolist()

    # token_txt = tokenizer.decode(indices[0]).strip().lower() TODO for debugging
    top_nodes = []
    for i in range(len(values)):
        top_nodes.append(DecodeNode(prob=values[i], token_idx=indices[i], 
                prev=node.prev+[node], prev_score=node.prev_score+[node.score]))
    return top_nodes

def baseline_beam_search(gen_model, inp_ids, tokenizer, args):
    bwidth = args["beam_size"]
    heap = [] # nodes to explore go here
    finished_hyps = [] # once a cand is done, put here
    dec_prefix = args['dec_prefix'] # decoding to get us started off

    # set up decoding from prefix
    last = None
    for prefix in dec_prefix:
        if last:
            starter = DecodeNode(prob=1, token_idx=prefix,
                                prev=[last], prev_score=[0])
        else:
            starter = DecodeNode(prob=1., token_idx=prefix,
                                prev=[], prev_score=[])
            last = starter

    heap.append(starter) # kick off decoding
    # keep on decoding until we hit ends for everything
    while len(finished_hyps)<args['beam_size']:
        # can't go over max_len
        if len(heap[0].prev)+1 == args["max_len"]:
            finished_hyps.extend(heap)
            break
        newnodes = []
        # keep track of all nodes
        for i in range(len(heap)):
            cur = heap.pop()
            if cur.token_idx == tokenizer.eos_token_id and len(cur.prev)>2:
                cur.finished = True
                finished_hyps.append(cur)
                bwidth-=1 # continue to expand only unfinished beams
            else:
                # get top beam-size nodes for each node in heap
                newnodes.extend(decode_step(cur, gen_model, inp_ids, tokenizer, args))
        # take top beamsize nodes with highest score
        nextbest = np.argsort(np.array([n.get_score_sum() for n in newnodes]))
        heap = [newnodes[int(ind)] for ind in nextbest[:bwidth]]      

    # sort hypotheses by average score and return
    finished_hyps.sort(key=lambda x: x.get_score_avg(), reverse=True)   
    return finished_hyps


# kick off baseline-decoding, given a gen model
def run_baseline_decoding(gen_model, source, tokenizer, args):
    input_ids = tokenizer(
            source, return_tensors="pt").input_ids.to(args['device'])
    # TODO record dec_calls for stats purposes
    results = baseline_beam_search(gen_model, input_ids, tokenizer, args)
    end_strs = [r.get_ending_str(tokenizer) for r in results]
    return results, end_strs
        
# kick off tfr-decoding, given a gen model, tfr
def run_tfr_decoding(gen_model, tfr_model, source, tokenizer, args):
    input_ids = tokenizer(
            source, return_tensors="pt").input_ids.to(args['device'])
    max_len = args['max_len'] # don't go over this
    dec_calls = 0 # track number of decoding calls

