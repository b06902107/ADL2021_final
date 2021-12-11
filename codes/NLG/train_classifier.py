import json
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset 
from transformers import AdamW, RobertaTokenizer, get_linear_schedule_with_warmup, RobertaForSequenceClassification, RobertaConfig
from tqdm import tqdm
import time
import os
import math
import sys

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

fp16_training = True

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
    good_count = bad_count = 0
    for dialogue in data:
        one_data = 'summarize: '
        for turn in dialogue['turns']:
            if turn['speaker'] == 'USER':
                one_data += 'USER : ' + turn['utterance'] + ' '
            elif turn['speaker'] == 'SYSTEM':
                #print(turn['frames'])
                if 'beginning' in turn.keys() and len(turn['beginning']) != 0:
                    for begin in turn['beginning']:
                        new_data.append({'dialog' : "".join(one_data), 'label' : begin['candidate'], 'status' : begin['label']})
                        if begin['label'] == 'good':
                            good_count += 1
                        elif begin['label'] == 'bad':
                            bad_count += 1

                one_data += 'SYSTEM : ' + turn['utterance'] + ' '

                if 'end' in turn.keys() and len(turn['end']) != 0:
                    for end in turn['end']:
                        new_data.append({'dialog' : "".join(one_data), 'label' : end['candidate'], 'status' : end['label']})
                        if end['label'] == 'good':
                            good_count += 1
                        elif end['label'] == 'bad':
                            bad_count += 1
    print('good = ', good_count, 'bad = ', bad_count)
    return new_data

data_dir = sys.argv[1]
train_data = data_select(read_data(os.path.join(data_dir, 'train')))
val_data = data_select(read_data(os.path.join(data_dir, 'dev')))

print(len(train_data))
print(len(val_data))
#print(train_data[0])

tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
config = RobertaConfig.from_pretrained('roberta-large')
config.num_labels = 1
config.problem_type = "multi_label_classification"
#print(config)

model = RobertaForSequenceClassification.from_pretrained('roberta-large', config=config)
#print(model.parameters)

class Chit_chat_Dataset(Dataset): 
    def __init__(self, mode, data):
        self.mode = mode
        self.data = data

    def __getitem__(self, index):
        if self.mode == 'train' or self.mode == 'val':
            source = tokenizer.encode_plus(
                    self.data[index]['dialog'],
                    self.data[index]['label'],
                    add_special_tokens=True,
                    max_length = 512,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask = True,
                    return_token_type_ids = True,
                    return_tensors = 'pt',
                )
            input_ids = source["input_ids"].squeeze()
            attn_mask = source["attention_mask"].squeeze()
            token_type_ids = source['token_type_ids'].squeeze()
            if self.data[index]['status'] == 'good':
                label = 1
            else:
                label = 0
            return input_ids, attn_mask, token_type_ids, torch.tensor(label), torch.tensor([label], dtype = torch.half)
        else:
            return input_ids, attn_mask

    def __len__(self):
        return (len(self.data))

train_set = Chit_chat_Dataset('train', train_data)
val_set = Chit_chat_Dataset('val', val_data)

train_batch_size = 8
train_dataloader = DataLoader(train_set, batch_size = train_batch_size, num_workers = 4, shuffle = True)
val_dataloader = DataLoader(val_set, batch_size = 32, num_workers = 4)

class Sigm(nn.Module):
    def __init__(self):
        super(Sigm, self).__init__()
        self.sig = nn.Sigmoid()
    def forward(self, x):
        y = self.sig(x)
        return y

grad_accum_steps = 4
num_epoch = 2
lr = 3e-5
logging_step = 500
val_step = 1000
update_steps_per_epoch = math.ceil(len(train_dataloader) / grad_accum_steps)
total_steps = update_steps_per_epoch * num_epoch
print('total_steps = ', total_steps)
optimizer = optim.AdamW(model.parameters(), lr = lr)
loss_function = nn.BCEWithLogitsLoss()
model = model.to(device)
Sig = Sigm().to(device)

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 400, num_training_steps = total_steps)
if fp16_training:
    model, optimizer, train_dataloader, loss_function, Sig = accelerator.prepare(model, optimizer, train_dataloader, loss_function, Sig) 

print("Start Training ...")
best_loss = np.inf
best_acc = 0
#for _ in range(0):
for epoch in range(num_epoch):
    step = 0
    train_loss = train_acc = 0
    count = correct = 0
    model.train()
    for data in tqdm(train_dataloader):
        step += 1
        data = [i.to(device) for i in data]
        #print(data[0].size())
        #print(data[1].size())
        #print(data[2].size())
        #print(data[4].size())
        outputs = model(input_ids=data[0], attention_mask=data[1], token_type_ids=data[2], labels=data[4])
        #print(outputs.loss)
        #print(outputs.logits)
        #logits2 = Sig(outputs.logits)
        #print(logits2)
        #logits = 1/(1 + torch.exp(-outputs.logits))
        loss = outputs.loss
        logits = outputs.logits
        #print(logits)
        #data[4] = data[4].unsqueeze(1)
        #print(data[4].size())
        #print(data[4])
        #print(loss)
        #loss2 = loss_function(logits, data[4])
        #print(loss2)
        #pred = torch.max(outputs.logits, 1)[1]
        #logits = Sig(logits)
        #print(logits)
        logits[logits>=0] = 1
        logits[logits<0] = 0
        #print(logits)
        #print(data[3])
        #pred = []
        #for logit in logits:
        #    if logit > 0.5:
        #        pred.append(1)
        #    else:
        #        pred.append(0)
        pred = logits

        train_loss += loss.item()
        #print(train_loss)
        if len(train_dataloader) % grad_accum_steps != 0 and len(train_dataloader) - step < grad_accum_steps:
            loss = loss / (len(train_dataloader) % grad_accum_steps)
        else:
            loss = loss / grad_accum_steps

        if fp16_training:
            accelerator.backward(loss)
        else:
            loss.backward()

        if step % grad_accum_steps == 0 or step == len(train_dataloader):       
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        for j in range(len(pred)):
            if pred[j] == data[3][j]:
                correct += 1
            count += 1

        if step % logging_step == 0:
            print(f"Epoch {epoch + 1} | Step {step} | acc = {correct / count:.3f} | loss = {train_loss / logging_step:.3f}")
            train_loss = train_acc = 0
            count = correct = 0

        if step % val_step == 0 or step % len(train_dataloader) == 0:
            print("Evaluating Dev Set ...")
            model.eval()
            with torch.no_grad():
                val_loss = val_acc = 0
                val_correct = val_count = 0
                for i, data in enumerate(tqdm(val_dataloader)):
                    data = [i.to(device) for i in data]
                    outputs = model(input_ids=data[0], attention_mask=data[1], token_type_ids=data[2], labels=data[4])
                    #print(outputs)
                    logits = outputs.logits
                    loss = outputs.loss
                    #data[4] = data[4].unsqueeze(1)
                    #loss = loss_function(logits, data[4])
                    #pred = torch.max(outputs.logits, 1)[1]
                    #logits = Sig(logits)
                    #print(logits)
                    logits[logits>=0] = 1
                    logits[logits<0] = 0
                    #print(logits)
                    pred = logits
                    #print(data[3])

                    #loss = outputs.loss
                    #print(loss)
                    #pred = torch.max(outputs.logits, 1)[1]
                    for j in range(len(pred)):
                        if pred[j] == data[3][j]:
                            val_correct += 1
                        val_count += 1
                    #print(loss)
                    val_loss += loss.item()
                    #print(val_loss)
                print(f"Validation | Epoch {epoch + 1} | acc = {val_correct / val_count:.3f} | loss = {val_loss / len(val_dataloader):.3f}")
                print(f"best_acc = {best_acc:.3f} | best_loss = {best_loss:.3f}")

                if val_correct / val_count > best_acc:
                    best_acc = val_correct / val_count
                    best_loss = val_loss / len(val_dataloader)
                    torch.save(model.state_dict(), "./model_classifer_new.ckpt")
            model.train()
