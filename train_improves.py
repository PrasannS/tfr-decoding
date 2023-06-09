import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        inp, hyp, label = row['inp'], row['hyp'], row['label']
        prompt = f'PROMPT: {inp} \n\n PARTIAL RESPONSE: {hyp}'

        inputs = self.tokenizer.encode_plus(
            prompt,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }

class T5BinaryClassifier(pl.LightningModule):
    def __init__(self, model_name, tokenizer, learning_rate, max_len=512):
        super().__init__()

        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.max_len = max_len

    def forward(self, input_ids, attention_mask, labels=None):
        if len(input_ids.shape)==1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
        if labels is not None:
            labels = labels.unsqueeze(-1)
            return self.model(input_ids, attention_mask=attention_mask, labels=labels)
        else:
            return self.model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=2)

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['label']
        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss
        self.log('train_loss', loss, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['label']
        logits = self(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=-1)
        
        acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
        # self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        #self.log("val_accuracy", acc, on_step=False, on_epoch=True, prog_bar=True)
    
        accuracy = (preds == labels).float().mean()
        self.log("val_accuracy", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        #self.log('val_accuracy', accuracy, sync_dist=True)
        return accuracy

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)

def balance_dataframe(dataframe):
    label_counts = dataframe['label'].value_counts()
    min_count = label_counts.min()
    balanced_data = dataframe.groupby('label').apply(lambda grp: grp.sample(min_count)).reset_index(drop=True)
    return balanced_data

def train_val_split(dataframe, test_size=0.1, random_state=42):
    unique_inp = dataframe['inp'].unique()
    train_inp, test_inp = train_test_split(unique_inp, test_size=test_size, random_state=random_state)
    
    train_df = dataframe[dataframe['inp'].isin(train_inp)].reset_index(drop=True)
    test_df = dataframe[dataframe['inp'].isin(test_inp)].reset_index(drop=True)
    
    return train_df, test_df

def train(dataframe, model_name='t5-small', epochs=2, batch_size=8, learning_rate=3e-5, max_len=512, val_interval=1):
    # Balance DataFrame and split into train and test
    dataframe = balance_dataframe(dataframe)
    train_df, test_df = train_val_split(dataframe)
    test_df.to_json("testimp.jsonl", lines=True, orient="records")

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    train_dataset = CustomDataset(train_df, tokenizer, max_len)
    test_dataset = CustomDataset(test_df, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=10)

    
    model = T5BinaryClassifier(model_name, tokenizer, learning_rate, max_len)

    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints",
        filename="{epoch:02d}-{val_accuracy:.4f}",
        save_top_k=2,
        monitor="val_accuracy",
        mode="max",
        save_last=True,
        save_weights_only=False,
        verbose=True,
    )
    trainer = pl.Trainer(
        max_epochs=epochs,
        # gpus=torch.cuda.device_count(),
        log_every_n_steps=val_interval,
        #check_val_every_n_epoch=val_interval,
        val_check_interval=500,
        #callbacks=[checkpoint_callback],
        enable_checkpointing=True,
        #early_stop_callback=None
    )
    trainer.fit(model, train_loader, val_loader)
    
def validate(dataframe, model_name='t5-small', epochs=2, batch_size=8, learning_rate=3e-5, max_len=512, val_interval=1):

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    test_dataset = CustomDataset(dataframe, tokenizer, max_len)

    val_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=10)

    
    model = T5BinaryClassifier(model_name, tokenizer, learning_rate, max_len)


    trainer = pl.Trainer(
        max_epochs=epochs,
        # gpus=torch.cuda.device_count(),
        log_every_n_steps=val_interval,
        #check_val_every_n_epoch=val_interval,
        val_check_interval=500,
        #callbacks=[checkpoint_callback],
        enable_checkpointing=True,
        #early_stop_callback=None
    )
    #trainer.fit(model, train_loader, val_loader, ckpt_path="./lightning_logs/version_3/checkpoints/epoch=0-step=2000.ckpt")
    trainer.validate(model, val_loader)
    

    


if __name__=="__main__":
    # Replace with your actual DataFrame
    inpdf = pd.read_json("std_dataset.jsonl", lines=True, orient="records")
    inpdf['hyp'] = inpdf['prefix']
    inpdf['label'] = (inpdf['sco']>.05).astype(int)


    # Train the model
    train(inpdf, model_name='stanfordnlp/SteamSHP-flan-t5-large', epochs=4, batch_size=8, learning_rate=3e-5, val_interval=1)