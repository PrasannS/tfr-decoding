import torch

@torch.no_grad()
def run_inference_step(model, input_ids, attention_mask=None, decoder_input_ids=None, targets=None, device='cuda:1', output_dec_hid=False, T=1):
    # we're using a standard setting
    decoder_input_ids = decoder_input_ids.unsqueeze(0).to(device)
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    model_inputs = {"input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "decoder_input_ids": decoder_input_ids,
                    }

    outputs = model(**model_inputs,
                    output_hidden_states=output_dec_hid,
                    use_cache=False, return_dict=True)
    # TODO figure out what's going on with targets

    # batch, dec seq, vocab size
    next_token_logits = outputs.logits[:, -1, :]
    if targets is not None:
        targets = targets.to(device)
        loss = torch.nn.functional.cross_entropy(
            input=next_token_logits, target=targets, reduction='none')
    else:
        loss = 0

    prob = torch.nn.functional.softmax(next_token_logits/T, dim=-1)
    # prob = next_token_logits.softmax(dim=-1)
    next_token = torch.argmax(next_token_logits, dim=-1)
    # next_token = next_token.unsqueeze(-1)
    next_token = next_token.tolist()    # confrim nested list?s
    # logging.info(f"Next token: {output}")
    # outputs['output'] = output
    return prob, next_token_logits, loss
