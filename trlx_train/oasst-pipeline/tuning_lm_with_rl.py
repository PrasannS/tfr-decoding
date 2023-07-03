import os

import torch
import time
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
import pandas as pd
from datasets import Dataset
from transformers import (
    Adafactor,
    AutoTokenizer,
    LlamaTokenizer,
    HfArgumentParser,
    pipeline,
    T5Tokenizer,
    T5ForConditionalGeneration
)

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

os.environ["WANDB_TAGS"] = "[\"llamatrl\"]"


tqdm.pandas()


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    # model was /home/prasann/Projects/tfr-decoding/apfarm_models/sft10k
    # also used lxuechen/tldr-gpt2-xl
    model_name: Optional[str] = field(default="/home/prasann/Projects/tfr-decoding/apfarm_models/sft10k", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default="", metadata={"help": "the tokenizer name"})
    reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    dataset_name: Optional[str] = field(default="eli5", metadata={"help": "the dataset name"})
    log_with: Optional[str] = field(default='wandb', metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    max_length: Optional[int] = field(default=512, metadata={"help": "maximum length for input"})
    output_max_length: Optional[int] = field(default=128, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=6, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="./checkpoints/tuning_llama_rl/",
                                      metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})
    samp_n: Optional[int] = field(default=1, metadata={"help": "how many samples from single input"})
    resamp_percent: Optional[float] = field(default=0.5, metadata={"help": "what percent of initial output toks to resample from"})


parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

set_seed(script_args.seed)

def adjust_input(strval):
    return "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n"+strval+"\n\n### Response:"

def adjust_tldr(inp):
    # remove initial information
    return inp[inp.index("POST:")+5:]
# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(tokenizer, dataset_name="eli5", input_min_text_length=2, input_max_text_length=8):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    # load imdb with datasets
    if dataset_name=="eli5":
        inpdf = pd.read_json('/home/prasann/Projects/tfr-decoding/trlx_train/oasst-pipeline/elidataset.jsonl', lines=True, orient='records').iloc[:10000]
        # adjust input to alpaca_farm format
        inpdf['history'] = [adjust_input(ihist) for ihist in inpdf['history']]
        # use 10k eli5 examples as starter dataset
        ds = Dataset.from_pandas(inpdf)
        ds = ds.rename_columns({"history": "review"})
    elif dataset_name=="tldr":
        ds = load_dataset("CarperAI/openai_summarize_tldr", split="train")
        ds = ds.select(range(100000))
        ds = ds.rename_columns({"prompt": "review"})
    else:
        ds = load_dataset(dataset_name, split="train")
        ds = ds.select(range(100000))
        #ds = ds.rename_columns({"text": "review"})
        ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    # input_size = LengthSampler(input_min_text_length, input_max_text_length)
    

    def tokenize(sample):
        # TODO trying out this thing for batching
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for question in sample["review"]:
            if dataset_name=="tldr":
                query = adjust_tldr(question)
            else:
                query = question
            #query = "Question: " + question + "\n\nAnswer: "
            tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    ds = ds.map(tokenize, batched=True)
    ds = ds.filter(lambda x: len(x["input_ids"]) < 512, batched=False)

    ds.set_format(type="torch")
    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

reward_model_name = script_args.reward_model_name
config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    batch_size=script_args.batch_size,
    mini_batch_size=script_args.mini_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    ppo_epochs=script_args.ppo_epochs,
    seed=script_args.seed,
    cliprange=0.3,
    cliprange_value=0.3,
    vf_coef=.1,
    horizon=10000,
    target=6,
    init_kl_coef=0.1,
    steps=20000,
    gamma=1,
    lam=0.95,
)

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
rw_kwargs = {
    "topk": None,
    "function_to_apply": "none",
    "batch_size": script_args.mini_batch_size,
    "truncation": True
}

if "decapoda" in script_args.model_name.lower():
    tokenizer = LlamaTokenizer.from_pretrained(script_args.model_name)
    # required for llama
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
            "pad_token": DEFAULT_PAD_TOKEN,
        }
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(tokenizer, script_args.dataset_name)

# Now let's build the model, the reference model, and the tokenizer.
current_device = Accelerator().local_process_index

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    script_args.model_name,
    load_in_8bit=True, # re-enable for llama model
    device_map={"": current_device},
    peft_config=lora_config,
)

optimizer = None
if script_args.adafactor:
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
    )

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)

# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug
   
if script_args.dataset_name=="eli5steam":
    steamtok = T5Tokenizer.from_pretrained('stanfordnlp/SteamSHP-flan-t5-xl')
    reward_model = T5ForConditionalGeneration.from_pretrained('stanfordnlp/SteamSHP-flan-t5-xl', device_map={"": current_device})
else: 
    # NOTE sub in sentiment classification to test things out
    reward_model = pipeline("text-classification", model="OpenAssistant/reward-model-deberta-v3-large-v2", device_map={"": current_device})

# TODO consider doing batching if this is slow
def steamscore(inpstr, ansstr):
    prompt = """POST: {context}

    RESPONSE A: {response}

    RESPONSE B: .

    Which response is better? RESPONSE
    """
    # get rid of \n's since steamshp seems to get weirded out by them
    formatted = prompt.format(context=inpstr.replace("\n", ""), response=ansstr.replace("\n", ""))
    x  = steamtok([formatted], return_tensors='pt').input_ids.to(device)
    outputs = reward_model.generate(x, return_dict_in_generate=True, output_scores=True, max_new_tokens=1)
    # get scores
    return torch.exp(outputs.scores[0][:, 71]) / torch.exp(outputs.scores[0][:,:]).sum(axis=1).item() # index 71 corresponds to the token for 'A'


sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": script_args.mini_batch_size, "truncation":True}

""" NOTE old rm pipeline
pipeline(
    "text-classification",
    model=reward_model_name,
    device_map={"": current_device},
    model_kwargs={"load_in_8bit": True},
    tokenizer=tokenizer,
)
"""

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    # "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
}
output_min_length = 60
output_max_length = script_args.output_max_length
output_length_sampler = LengthSampler(output_min_length, output_max_length)

first = True
for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    if first: 
        print(tokenizer.batch_decode(query_tensors))
    model.gradient_checkpointing_disable()
    model.pretrained_model.config.use_cache = True
    with torch.no_grad():
        # Get response from Causal LM
        response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            length_sampler=output_length_sampler,
            batch_size=script_args.mini_batch_size,
            **generation_kwargs,
        )
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
    """
    Non-batched version of stuff: 
    
    response_tensors = []
    print("queries in this batch ", len(query_tensors))
    for query in query_tensors:
        if first:
            start = time.time()
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len
        response = ppo_trainer.generate(query, **generation_kwargs)
        response_tensors.append(response.squeeze()[-gen_len:])
        if first:
            print("elapsed ", time.time()-start)
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
    """
    

    if first: 
        print(batch['response'])
        first = False
    if script_args.dataset_name=="eli5steam":
        rewards = [steamscore(q, r) for q, r in zip(batch["query"], batch["response"])]
    else:
        # Compute sentiment score
        texts = [q + "[SEP]" + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = reward_model(texts, **sent_kwargs)
        # TODO validate that this is ok
        rewards = [torch.tensor(output[0]["score"]) for output in pipe_outputs]

    # Run PPO step
    model.gradient_checkpointing_enable()
    model.pretrained_model.config.use_cache = False

    torch.cuda.empty_cache()
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)
        
    try:
        if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
            ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")
    except:
        print("loading to hf failed")
