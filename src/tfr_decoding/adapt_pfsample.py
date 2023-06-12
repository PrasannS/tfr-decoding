# do monkey patching with huggingface sample method

from typing import Optional, Union, List
import warnings

import torch
import torch.distributed as dist
from torch import nn
import numpy as np
import random

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

def get_checkpoint(rec, rec_int, prevcheck):
    retind = -1
    for i in range(len(rec)): # take last value where prediction isn't 0
        if rec[i]>0.5:
            retind = i
    if retind==-1:
        return prevcheck
    else:
        return (retind+1)*rec_int
    
def rec_dist(rec, curind):
    # resample based on how many ones there are
    prop = (sum(rec)/len(rec))
    if prop==1:
        return 0
    return 1-prop/(30/min(curind, 30))

def earlymax(lis):
    mind = 0
    mval = -10000
    for i in range(len(lis)):
        if lis[i] > mval:
            mind = i
            mval = lis[i]
    return mind

def mass_sco(record, decay):
    lscos = []
    for i in range(len(record)):
        csco = 0
        # to start, only look at values from left (including self)
        for j in range(i+1):
            # subtract if bad
            if record[j]==0:
                # larger penalty the closer we get to index in question
                csco = csco-pow(decay, i-j)
            else:
                # bigger reward if we're closer to given index
                csco = csco+pow(decay, i-j)
        lscos.append(csco)
    # TODO for debugging
    print(lscos)
    return earlymax(lscos)
                
# monkeypatched hyper-params: rec_n, check_n, cont_len
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
    
    print("monkeysamp")

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
    tokstats = {
        "entropy":[],
    }
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )
    rec_n = self.rec_n # record every this many tokens
    if hasattr(self, "source_str"):
        source_str = self.source_str
    # record of how good things are so far
    record = []
    # at end of decoding, choose best prefix to resample from. After external check, we can just recall as adaptive baseline code as necessary
    if hasattr(self, "over_inpids"):
        if self.over_inpids is not None:
            input_ids = self.over_inpids
            record = self.oldrecord
    # TODO setup later
    stdrecord = []
    unused_toks = 0
    checkpoint = 1
    resinds = []

    # keep track of which sequences are already finished
    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)

    this_peer_finished = False  # used by synced_gpus only
    lastoutputs = None
    # auto-regressive generation
    while True:
        self.decoded_toks = self.decoded_toks + 1
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        if lastoutputs is None: # not sure if this saves anything
            lastoutputs = outputs

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
        tokstats['entropy'].append(entropy)
                
        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        
        
        # we need to do prefix sampling check
        if input_ids.shape[1]%rec_n==0:
            hyp_str = self.tok.decode(input_ids[0], skip_special_tokens=True)
            goodlogits = self.qualitypref.predsingle(source_str, hyp_str)
            pred = torch.argmax(goodlogits, dim=-1)
            conf = torch.max(goodlogits, dim=-1).values # TODO getting logits might be a bit weird
            record.append(pred)
        if input_ids.shape[1]%15==0:
            print("-")
                
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id is not None:
            unfinished_sequences = unfinished_sequences.mul((sum(next_tokens != i for i in eos_token_id)).long())

        # stop when each sentence is finished, or if we exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
            if not synced_gpus:
                break
            else:
                this_peer_finished = True
    
    DECAY = self.decay
    print(record)
    # otherwise go by weighted function
    checkpt = mass_sco(record, DECAY)
    # if mostly 1s at the end, try again from the middle
    if checkpt == len(record)-1:
        checkpt = int(len(record)/2)
    if checkpt==0 and record[0]==0:
        self.over_inpids = None # if a bunch of 0s at the beginning, just resample whole thing
    else:
        # prepare for next round of decoding
        self.over_inpids = input_ids[:, :checkpt*rec_n]
        # use record from previous decoding
        self.oldrecord = record[:checkpt+1]
    
    # make it all in batch form, TODO so far batching not supported
    if input_ids.shape[0]>8:
        input_ids = input_ids.unsqueeze(0)

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return SampleEncoderDecoderOutput(
                sequences=input_ids,
                # we override normal scores output with token stats dictionary
                scores=tokstats,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return SampleDecoderOnlyOutput(
                sequences=input_ids,
                scores=tokstats,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return input_ids