# do monkey patching with huggingface sample method

from typing import Optional, Union, List
import warnings

import torch
import torch.distributed as dist
from torch import nn
import numpy as np

from transformers.generation.logits_process import (
    LogitsProcessorList,
)

from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)

from transformers.utils import logging
from transformers.generation.utils import SampleOutput, SampleDecoderOnlyOutput, SampleEncoderDecoderOutput


logger = logging.get_logger(__name__)

CTHRESH = 0.75
TTHRESH = 0.85

# score a single example (I don't think there's enough space to batch this?)
def get_reward_single(self, inpdict):
    template = "POST: {context:s} \n\nRESPONSE A:{hyp:s} \n\nRESPONSE B: .\n\n Which response is better? RESPONSE "
    inp = template.format(context=inpdict['context'], hyp=inpdict['hyp'])
    x = self.downtok([inp], return_tensors='pt').input_ids.to(self.downmetric.device)
    outputs = self.downmetric.generate(x, return_dict_in_generate=True, output_scores=True, max_new_tokens=1)
    return torch.exp(outputs.scores[0][:, 71]) / torch.exp(outputs.scores[0][:,:]).sum(axis=1).item() # index 71 corresponds to the token for 'A'

# HPARAMS
# rec_n, cont_checks, source_str, checklist, 
def sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: Optional[bool] = False,
    **model_kwargs,
) -> Union[SampleOutput, torch.LongTensor]:
    
    if True:
        print("start decoding")

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = ()
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )
    
    # list of points (# of decoded tokens) where we need to do a check / potentially backtrack
    if hasattr(self, "checklist"):
        CHECKS = [1]+self.checklist+[10000]
        rec_n = self.rec_n
        source_str = self.source_str
        cont_checks = self.cont_checks
    else:
        # never resample
        CHECKS = [1, 10000]
        rec_n = 3
        cont_checks = 3
    
    cind = 1 # which checkpoint we're on
    resamps = 0 # how many times we've resampled (to manage budget)
    self.decoded_toks = 0

    cur_probs = [] # working list of probs for stuff we've seen (for stuck case)
    cur_inpids = [] # working list of inpids for stuff we've tried (for stuck case)
    #cur_outstates = [] # working list of output states to recompute kwargs
    next_probs = [] # to carry over
    next_inpids = [] # to carry over      
    #next_outstates = [] # to carry over  
    
    record = [] # record of adjusted probability so far
    max_indiv_resamps = 8 # how much can we be stuck at single point
    
    # keep track of which sequences are already finished
    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)

    this_peer_finished = False  # used by synced_gpus only
    isresamp = False

    fouts = []
    fscores = []
    fstrs=  []
    # auto-regressive generation
    while True:
        if True: # allow vscode folding on this
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            self.decoded_toks = self.decoded_toks+1
            if isresamp:
                # a little inefficient
                model_kwargs['past_key_values']=None
                model_kwargs['use_cache']=None
                # prepare model inputs
                model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            else:
                # prepare model inputs
                model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            
            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            # get entropy
            entropy = probs*torch.log(probs).nan_to_num()
            entropy = -1*torch.sum(entropy, dim=1)
                    
            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                
        # NOTE: code to update record
        if input_ids.shape[1]%rec_n==0 and resamps<self.max_resamps and cind<(len(CHECKS)-1): # TODO add something to reduce calls to save compute
            # convert stuff to strings, back
            hyp_str = self.tok.decode(input_ids[0], skip_special_tokens=True)
            print(hyp_str)
            pref_out = self.qualitypref.predsingle(source_str, hyp_str, True)
            # get adjusted probability
            transition_scores = self.qualitypref.model.compute_transition_scores(
                pref_out.sequences, pref_out.scores, normalize_logits=True
            )
            pred = float(np.argmax(pref_out.scores[-1].cpu()))
            prob = float(np.exp(transition_scores[0][-1].cpu())) 
            if pred==0:
                prob = 1-prob
            # store
            record.append(prob)
        
        isresamp = False
        # we need to do prefix sampling check, TODO logic to just make it adaptive baseline
        if input_ids.shape[1]%CHECKS[cind]==0 and resamps<self.max_resamps:
            curprob = max(record[-1*cont_checks:])
            print("checkpoint prob is ", curprob)
            
            # we're good to keep sampling
            if curprob>CTHRESH:
                cind = cind+1
                next_probs, next_inpids = [], []
            else:
                print(cur_probs)
                next_probs.append(curprob)
                next_inpids.append(input_ids)
                # we've been stuck at this checkpoint too much
                if len(next_probs)==max_indiv_resamps:
                    cind = len(CHECKS)-1 # just switch to adaptive with what we have / ordered by prefix sampling
                    # manual starting points
                    cur_probs, cur_inpids = next_probs, next_inpids
                    next_inpids, next_probs = [], []
                
                input_ids = input_ids[:, :CHECKS[cind-1]]
                resamps = resamps+1
                if len(cur_probs)>0: # instead start from best scoring thing we left off with
                    nextind = int(np.argmax(cur_probs))
                    input_ids = cur_inpids.pop(nextind)
                    cur_probs.pop(nextind)
                record = record[:-1*cont_checks] # reset record (not technically needed)
                isresamp = True
                
        if True:
            # not sure if this is the right thing to do? 

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((sum(next_tokens != i for i in eos_token_id)).long())
                
            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    # treat this as adaptive baseline, TODO multi-gpu code not ready
                    out = self.tokenizer.batch_decode(input_ids)[0]
                    score = get_reward_single(self, {'context':source_str, 'hyp':out}).cpu()
                    fouts.append(input_ids)
                    fscores.append(score)
                    fstrs.append(out)
                    print("adaptive, score ", score, " ", out)
                    # we don't need to sample anymore
                    if score>TTHRESH:
                        break
                    if len(fscores)<4 and len(cur_probs)>0: # instead start from best scoring thing we left off with
                        nextind = int(np.argmax(cur_probs))
                        input_ids = cur_inpids.pop(nextind)
                        cur_probs.pop(nextind)
                        unfinished_sequences = torch.tensor(1).to(self.device)
                    else:
                        break
                else:
                    this_peer_finished = True
    
    self.allscos = fscores
    self.allstrs = fstrs
    # we ended up doing semi-adaptive backup, use best option
    if len(fscores)>0:
        input_ids = fouts[int(np.argmax(fscores))]
        
    if True:
        # make it all in batch form
        if input_ids.shape[0]>8:
            input_ids = input_ids.unsqueeze(0)

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return SampleEncoderDecoderOutput(
                    sequences=input_ids,
                    # we override normal scores output with token stats dictionary
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return SampleDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids