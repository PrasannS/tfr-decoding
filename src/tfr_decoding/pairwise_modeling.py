import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
import numpy as np
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
        self.predictions = []
        self.labels = []
        self.probarrs = []
        #self.linear = torch.nn.Linear(self.model.config.d_model, feature_size)
        
        # make single prediction given inputs
    def predsingle(self, inp, hypa, hypb, logits=False):
        inp = self.getinp(inp, hypa, hypb)
        input_ids = inp['input_ids']
        attention_mask = inp['attention_mask']
        if len(input_ids.shape)==1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
        # A, B
        CLASSES = [71, 272]
        if logits:
            out = self.model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=1, return_dict_in_generate=logits,output_scores=logits)
            # also get probs
            cprobs = []
            for c in CLASSES:
                out.sequences[0][1] = c
                transition_scores = self.model.compute_transition_scores(
                    out.sequences, out.scores, normalize_logits=True
                )
            cprobs.append(float(np.exp(transition_scores[0][0].cpu())))
            return int(np.argmax(cprobs)), cprobs
        return self.model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=1)
    
    # make input for prediction
    def getinp(self, inp, hypa, hypb):

        prompt = f'Does partial response A or B answer the question best?\n\n QUESTION: {inp} \n\n PARTIAL RESPONSE A: {hypa} \n\n PARTIAL RESPONSE B: {hypb}\n\n The better response is RESPONSE '

        inputs = self.tokenizer(
            prompt,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs.input_ids.to(self.device),
            'attention_mask': inputs.attention_mask.to(self.device),
        }


    def forward(self, input_ids, attention_mask, labels=None):
        if len(input_ids.shape)==1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
                
        if labels is not None:
            labels = labels.unsqueeze(-1)
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)

            return outputs
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
    
    def validation_step(self, batch, batch_idx):
        
        input_ids, attention_mask, labels = batch
        
        CLASSES = [71, 272]
        out = self.model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=1, return_dict_in_generate=True,output_scores=True)
        preds = out.sequences[:, -1].tolist()
        self.predictions.extend(preds)
        self.labels.extend(labels.tolist())
        # also get probs
        cprobs = []
        for c in CLASSES:
            out.sequences[:, 1] = c
            transition_scores = self.model.compute_transition_scores(
                out.sequences, out.scores, normalize_logits=True
            )
            cprobs.append(np.exp(transition_scores[:, 0].cpu()))
            
        cprobs = torch.stack(cprobs).T.tolist()
        self.probarrs.extend(cprobs)
        #print(logits)
        
        
        acc = accuracy_score(labels.cpu().numpy(), preds)
        # self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        #self.log("val_accuracy", acc, on_step=False, on_epoch=True, prog_bar=True)
    
        self.log("val_accuracy", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        #self.log('val_accuracy', accuracy, sync_dist=True)
        return acc

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)
    
def validate(dataframe, ckpt_path, model_name='stanfordnlp/SteamSHP-flan-t5-large', epochs=2, batch_size=8, learning_rate=3e-5, max_len=512, val_interval=1):

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    test_dataset = CustomDataset(dataframe, tokenizer, max_len)

    val_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, collate_fn=custom_collate)

    model = T5BinaryClassifier(model_name, tokenizer, learning_rate, max_len)

    trainer = pl.Trainer(
        max_epochs=epochs,
        devices=1,
        # gpus=torch.cuda.device_count(),
        log_every_n_steps=val_interval,
        #check_val_every_n_epoch=val_interval,
        val_check_interval=500,
        #callbacks=[checkpoint_callback],
        enable_checkpointing=True,
        #early_stop_callback=None
    )
    #trainer.fit(model, train_loader, val_loader, ckpt_path="./lightning_logs/version_3/checkpoints/epoch=0-step=2000.ckpt")
    trainer.validate(model, val_loader, ckpt_path=ckpt_path)  
    
    return model.predictions, model.labels, model.probarrs

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
    testdf = pd.read_json("output/testsetlarge.jsonl", orient='records', lines=True)
    preds, labels, probs = validate(testdf, "lightning_logs/pairmodel/checkpoints/epoch=4-step=52074.ckpt")