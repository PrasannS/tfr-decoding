import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config, Trainer, TrainingArguments
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import Dataset

# Load your DataFrame
df = pd.read_json("output/traingoodclassif.json", lines=True, orient="records")

thresh = 0.85
df['label'] = (df['sco'] > thresh).astype(int)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

class_weights = compute_class_weight('balanced', classes=np.unique(train_df['label']), y=train_df['label'])
class_weights = torch.tensor(class_weights).to(torch.float32)

def compute_weighted_loss(model, inputs, class_weights):
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logits = outputs.logits
    loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
    loss = loss_fct(logits.view(-1, 2), labels.view(-1))
    return loss

model_name = 'stanfordnlp/SteamSHP-flan-t5-xl'
tokenizer = T5Tokenizer.from_pretrained(model_name)
config = T5Config.from_pretrained(model_name, num_labels=2)
model = T5ForConditionalGeneration.from_pretrained(model_name, config=config)

MAXLEN = 512

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        def __getitem__(self, idx):
            example = self.df.iloc[idx]
            tokenized_input = self.tokenizer(
                f"PROMPT: {example['inp']} \n\n PARTIAL ANSWER: {example['hyp']}",
                truncation=True,
                padding='max_length',
                max_length=self.maxlen,
                return_tensors="pt"
            )

            if len(tokenized_input['input_ids']) > self.maxlen:
                tokenized_input['input_ids'] = tokenized_input['input_ids'][:, :self.maxlen]
                tokenized_input['attention_mask'] = tokenized_input['attention_mask'][:, :self.maxlen]

            # Add the label
            tokenized_input["labels"] = torch.tensor(example["label"], dtype=torch.long)

            print(f"Tokenized input: {tokenized_input}")

            return tokenized_input
    
class CustomTrainer(Trainer):
    def compute_loss(model, inputs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        return loss

train_dataset = CustomDataset(train_df, tokenizer, MAXLEN)
test_dataset = CustomDataset(test_df, tokenizer, MAXLEN)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy='epoch',
    logging_dir='./logs',
    logging_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    greater_is_better=True,
    seed=42,
    #n_gpu=2,  
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=lambda eval_preds: {'accuracy': (eval_preds.predictions.argmax(axis=-1) == eval_preds.label_ids).mean()},
    #compute_loss=compute_weighted_loss
)

trainer.train()
eval_results = trainer.evaluate()
print(eval_results)