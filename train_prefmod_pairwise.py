import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from torch.nn import functional as F
from torch.nn.modules.distance import PairwiseDistance
from pytorch_lightning.strategies.ddp import DDPStrategy
from torch.nn.utils.rnn import pad_sequence


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        inp, hypa, hypb, label = row['inp'], row['hyp_a'], row['hyp_b'], row['label']
        prompt = f'Does partial response A or B answer the question best?\n\n QUESTION: {inp} \n\n PARTIAL RESPONSE A: {hypa} \n\n PARTIAL RESPONSE B: {hypb}\n\n The better response is RESPONSE '

        inputs = self.tokenizer(
            prompt,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        return inputs['input_ids'], inputs['attention_mask'], torch.tensor(label)


class T5BinaryClassifier(pl.LightningModule):
    def __init__(self, model_name='stanfordnlp/SteamSHP-flan-t5-large', learning_rate=3e-5, max_len=512, feature_size=256):
        super().__init__()

        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.learning_rate = learning_rate
        self.max_len = max_len
        #self.linear = torch.nn.Linear(self.model.config.d_model, feature_size)


    def forward(self, input_ids, attention_mask, labels=None):
        if len(input_ids.shape)==1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
                
        if labels is not None:
            labels = labels.unsqueeze(-1)
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            #features = self.linear(outputs.encoder_last_hidden_state)
            # train time
            
            return outputs#, features.mean(dim=-2)
            #return outputs, None
        else:
            # test time
            return self.model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=1)

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        if batch_idx==0:
            print(self.tokenizer.batch_decode(input_ids))
        #try: 
            #input_ids, attention_mask, labels = batch
        
        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss

        self.log('train_loss', loss, sync_dist=True)
        return loss
        #except:
        #    print("Strange issue occurred")
        #    return None
    
    def validation_step(self, batch, batch_idx):
        
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        #print(logits)
        preds = logits[:, -1]
        
        acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
        # self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        #self.log("val_accuracy", acc, on_step=False, on_epoch=True, prog_bar=True)
    
        self.log("val_accuracy", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        #self.log('val_accuracy', accuracy, sync_dist=True)
        return acc

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)

def train(dataframe, model_name='t5-small', epochs=2, batch_size=8, learning_rate=3e-5, max_len=512, val_interval=1):
    # Balance DataFrame and split into train and test
    test_df = pd.read_json("output/testsetbalanced.jsonl", lines=True, orient="records")
    # crafted to not include stuff from earlier training
    train_df = pd.read_json("output/trainsetbalanced.jsonl", lines=True, orient="records")
    
    print("TRAIN SET SIZE IS ", len(train_df))
    print(train_df['label'].iloc[:20])
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    # even number of batches per device?
    train_df= train_df.iloc[:int(len(train_df)/24)*24]
    test_df = test_df.iloc[:int(len(test_df)/24)*24]
    train_dataset = CustomDataset(train_df, tokenizer, max_len)
    test_dataset = CustomDataset(test_df, tokenizer, max_len)
    
    print(len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6, collate_fn=custom_collate)
    print(len(train_loader))
    val_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, collate_fn=custom_collate)

    
    model = T5BinaryClassifier(model_name, learning_rate, max_len)
    # contrastive model
    # model = T5BinaryClassifier.load_from_checkpoint('lightning_logs/bestmodel2/checkpoints/epoch=2-step=36436.ckpt')
    print(torch.cuda.device_count())
    print(epochs)
    trainer = pl.Trainer(
        max_epochs=epochs+1,
        min_epochs=epochs,
        strategy = DDPStrategy(find_unused_parameters=False),
        accelerator = 'gpu',
        devices = -1,
        log_every_n_steps=1,
        #check_val_every_n_epoch=val_interval,
        val_check_interval=0.1,
        #find_unused_parameters=False,
        #callbacks=[checkpoint_callback],
        enable_checkpointing=True,
        #fast_dev_run=True,
        #precision=16,
        gradient_clip_val=1.0,  # Optional: gradient clipping
        #resume_from_checkpoint=,
        #early_stop_callback=None
    )
    trainer.fit(model, train_loader, val_loader)
    
def custom_collate(batch):
    input_ids = [item[0].squeeze() for item in batch]
    attention_mask = [item[1].squeeze() for item in batch]
    labels = [item[2] for item in batch]

    # Pad sequences to the max length in the batch
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    labels = torch.stack(labels)

    return input_ids, attention_mask, labels


if __name__=="__main__":
    # Replace with your actual DataFrame
    # inpdf = pd.read_json("output/largerpfmdataset.jsonl", lines=True, orient="records")


    # Train the model
    train(None, model_name='stanfordnlp/SteamSHP-flan-t5-large', epochs=4, batch_size=8, learning_rate=3e-5, val_interval=1)