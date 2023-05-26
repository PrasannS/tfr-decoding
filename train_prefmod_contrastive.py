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

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.groups = []
        for _, d in dataframe.groupby('inp'):
            if len(d)==6:
                self.groups.append(d)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        group = self.groups[index]
        prompts = []
        labels = []
        for _, row in group.iterrows():
            inp, hyp, label = row['inp'], row['hyp'], row['label']
            prompt = f'PROMPT: {inp} \n\n PARTIAL RESPONSE: {hyp}'
            prompts.append(prompt)
            labels.append(label)

        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        return inputs['input_ids'], inputs['attention_mask'], torch.tensor(labels)


class T5BinaryClassifier(pl.LightningModule):
    def __init__(self, model_name='stanfordnlp/SteamSHP-flan-t5-large', learning_rate=3e-5, max_len=512, feature_size=256):
        super().__init__()

        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.learning_rate = learning_rate
        self.max_len = max_len
        self.linear = torch.nn.Linear(self.model.config.d_model, feature_size)


    def forward(self, input_ids, attention_mask, labels=None):
        if len(input_ids.shape)==1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
                
        if labels is not None:
            labels = labels.unsqueeze(-1)
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            features = self.linear(outputs.encoder_last_hidden_state)
            # train time
            
            return outputs, features.mean(dim=-2)
            #return outputs, None
        else:
            # test time
            return self.model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=2)

    # TODO reformulate using vera's loss
    def contrastive_loss(self, features, labels, temp=0.05):
        # get cosine sim for all possibilities
        similarity = torch.nn.functional.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)
        # only keep unique pairs, perform vera operation
        similarity = torch.exp(similarity)/temp
        similarity = similarity - torch.triu(similarity)
        # matrix of which labels are equal for which inds
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        
        positive_similarity = (similarity*pos_mask).sum()
        
        # final loss, if all positive labels then no loss, if diffeerent stuff super similar then high loss
        return -1*torch.log(positive_similarity/similarity.sum())
    
    def process_batch(self, batch):
        bsize = len(batch[0]['input_ids'])*len(batch)
        input_ids = torch.cat([b['input_ids'] for b in batch], dim=1).reshape([bsize, 512])
        attention_mask = torch.cat([b['attention_mask'] for b in batch], dim=1).reshape([bsize, 512])
        labels = []
        for i in range(len(batch[0]['input_ids'])):
            for j in range(len(batch)):
                labels.append(batch[j]['label'][i])
        labels = torch.stack(labels)
        return input_ids.long(), attention_mask.long(), labels.int()

    def training_step(self, batch, batch_idx):
        try: 
            input_ids, attention_mask, labels = batch
        
            outputs, features = self(input_ids, attention_mask, labels)
            loss = outputs.loss
            contrastive_loss = 0
            batch_size = int(len(input_ids)/GSIZE)
            # run contrastive loss on each item in group
            # TODO temporarily remove contrastive loss
            for i in range(batch_size):
               contrastive_loss = contrastive_loss+self.contrastive_loss(features[i*GSIZE:(i+1)*GSIZE, :], labels[i*GSIZE:(i+1)*GSIZE])
            total_loss = loss + contrastive_loss
            self.log('train_loss', total_loss, sync_dist=True)
            self.log('contrastive_loss', contrastive_loss, sync_dist=True)
            return total_loss
        except:
            print("Strange issue occurred")
            return None
    
    def validation_step(self, batch, batch_idx):
        
        input_ids, attention_mask, labels = batch
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
    #dataframe = balance_dataframe(dataframe)
    #train_df, test_df = train_val_split(dataframe)
    #test_df.to_json("pftest2.jsonl", lines=True, orient="records")
    #train_df.to_json("pftrain2.jsonl", lines=True, orient="records")
    test_df = pd.read_json("output/testsetlarge.jsonl", lines=True, orient="records")
    # crafted to not include stuff from earlier training
    train_df = pd.read_json("output/trainsetlarge.jsonl", lines=True, orient="records")
    
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
    model = T5BinaryClassifier.load_from_checkpoint('lightning_logs/bestmodel2/checkpoints/epoch=2-step=36436.ckpt')
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

GSIZE = 6
def custom_collate(original_batch):
    lens = list([o[0].shape[1] for o in original_batch])
    inpids = torch.zeros(GSIZE*len(original_batch), max(lens))
    amasks = torch.zeros(GSIZE*len(original_batch), max(lens))
    try: 
        # fit batches together (different sizes are ok I think?)
        for i in range(len(original_batch)):
            inpids[GSIZE*i:GSIZE*(i+1), :lens[i]] = original_batch[i][0]
            amasks[GSIZE*i:GSIZE*(i+1), :lens[i]] = original_batch[i][1]
    except:
        print("Strange Data Error")
        tmplabels = torch.zeros(len(inpids))
        curlabels = torch.cat([o[2] for o in original_batch], dim=0).long()
        tmplabels[:len(curlabels)] = curlabels
        return inpids.long(), amasks.long(), curlabels.long()

    # get batched up stuff
    return inpids.long(), amasks.long(), torch.cat([o[2] for o in original_batch], dim=0).long()

if __name__=="__main__":
    # Replace with your actual DataFrame
    inpdf = pd.read_json("output/largerpfmdataset.jsonl", lines=True, orient="records")


    # Train the model
    train(inpdf, model_name='stanfordnlp/SteamSHP-flan-t5-large', epochs=4, batch_size=1, learning_rate=3e-5, val_interval=1)