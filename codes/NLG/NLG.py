import json
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup, T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
import time
import os
import sys

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

fp16_training = False

if fp16_training:
    #!pip install accelerate==0.2.0
    from accelerate import Accelerator
    accelerator = Accelerator(fp16=True)
    device = accelerator.device

random.seed(811)
np.random.seed(811)
torch.manual_seed(811)
torch.cuda.manual_seed(811)
torch.cuda.manual_seed_all(811)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def read_data(data_dir):
    data = []
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        with open(file_path, 'r', encoding="utf-8") as reader:
            data.extend(json.load(reader))
    return data

def data_select(data):
    new_data = []
    for dialogue in data:
        one_data = 'summarize: '
        for turn in dialogue['turns']:
            if turn['speaker'] == 'USER':
                one_data += 'USER : ' + turn['utterance'] + ' '
            elif turn['speaker'] == 'SYSTEM':
                #print(turn['frames'])
                if 'beginning' in turn.keys() and len(turn['beginning']) != 0:
                    for begin in turn['beginning']:
                        if begin['label'] == 'good':
                            new_data.append({'dialog' : "".join(one_data), 'label' : begin['candidate']})

                one_data += 'SYSTEM : ' + turn['utterance'] + ' '

                if 'end' in turn.keys() and len(turn['end']) != 0:
                    for end in turn['end']:
                        if end['label'] == 'good':
                            new_data.append({'dialog' : "".join(one_data), 'label' : end['candidate']})
    return new_data

data_dir = sys.argv[1]
train_data = data_select(read_data(os.path.join(data_dir, 'train')))
val_data = data_select(read_data(os.path.join(data_dir, 'dev')))

#print(len(train_data))
#print(len(val_data))
#print(train_data[0])

tokenizer = T5Tokenizer.from_pretrained("t5-base")
#special_tokens_dict = {'bos_token': '<s>'}
#tokenizer.add_special_tokens(special_tokens_dict)

model = T5ForConditionalGeneration.from_pretrained("t5-base")
#model.resize_token_embeddings(len(tokenizer))


class Chit_chat_Dataset(Dataset): 
    def __init__(self, mode, data):
        self.mode = mode
        self.data = data

    def __getitem__(self, index):
        source = tokenizer.batch_encode_plus([self.data[index]['dialog']], max_length=512, padding='max_length', truncation=True, return_tensors="pt")
        input_ids = source["input_ids"].squeeze()
        attn_mask = source["attention_mask"].squeeze()

        if self.mode == 'train' or self.mode == 'val':
            self.targets = tokenizer.batch_encode_plus([self.data[index]['label']], max_length=50, padding='max_length', truncation=True, return_tensors="pt")
            target_input_ids = self.targets["input_ids"].squeeze()
            target_attn_mask = self.targets["attention_mask"].squeeze()

            return input_ids, attn_mask, target_input_ids, target_attn_mask
        else:
            return input_ids, attn_mask

    def __len__(self):
        return (len(self.data))

train_set = Chit_chat_Dataset('train', train_data)
val_set = Chit_chat_Dataset('val', val_data)

train_batch_size = 16
train_dataloader = DataLoader(train_set, batch_size = train_batch_size, num_workers = 4, shuffle = True)
val_dataloader = DataLoader(val_set, batch_size = 16, num_workers = 4)

num_epoch = 10
lr = 3e-5
logging_step = 150
val_step = 300
total_steps = len(train_dataloader) * num_epoch
#print(total_steps)
optimizer = optim.AdamW(model.parameters(), lr = lr)
model = model.to(device)

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 350, num_training_steps = total_steps)
if fp16_training:
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader) 

print("Start Training ...")
best_loss = np.inf
#for _ in range(0):
for epoch in range(num_epoch):
    step = 0
    train_loss = 0
    model.train()
    for data in tqdm(train_dataloader):
        data = [i.to(device) for i in data]
        labels = data[2]
        #print(data[0].size())
        #print(data[1].size())
        #print(data[2].size())
        #print(data[3].size())
        #print(labels)
        labels[labels[:, :] == tokenizer.pad_token_id] = -100
        #print(labels)

        outputs = model(input_ids=data[0], attention_mask=data[1], labels = labels)
        #print(outputs)
        loss = outputs.loss
        #print(loss)

        optimizer.zero_grad()
        if fp16_training:
            accelerator.backward(loss)
        else:
            loss.backward()        
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()

        step += 1
        if step % logging_step == 0:
            print(f"Epoch {epoch + 1} | Step {step} | loss = {train_loss / logging_step:.3f}")
            train_loss = 0

        if step % val_step == 0 or step % len(train_dataloader) == 0:
            print("Evaluating Dev Set ...")
            model.eval()
            predict = []
            with torch.no_grad():
                val_loss = 0
                for i, data in enumerate(tqdm(val_dataloader)):
                    data = [i.to(device) for i in data]
                    labels = data[2]
                    labels[labels[:, :] == tokenizer.pad_token_id] = -100
                    outputs = model(input_ids=data[0], attention_mask=data[1], labels = labels)
                    val_loss += outputs.loss.item()
                print(f"Validation | Epoch {epoch + 1} | loss = {val_loss / len(val_dataloader):.3f} | best loss = {best_loss:.3f}")

                if val_loss / len(val_dataloader) < best_loss:
                    #result = []
                    #for i, data in enumerate(val_data):
                    #    result.append({'id': val_data[i]['id'], 'title': predict[i]})
                    #with open('public_result.jsonl', 'w') as f:
                    #    for entry in result:
                    #        json.dump(entry, f)
                    #        f.write('\n')
                    #best_rouge = rouge_1
                    best_loss = val_loss / len(val_dataloader)
                    torch.save(model.state_dict(), "./model_NLG.ckpt")
            model.train()