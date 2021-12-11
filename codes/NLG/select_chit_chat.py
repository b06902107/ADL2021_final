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

def read_data(file):
    with open(file, 'r', encoding="utf-8") as reader:
        data = json.load(reader)
    return data

#train_data = data_select(read_data('./data-0614/train'))
#val_data = data_select(read_data('./data-0614/dev'))
test_seen_data = read_data(sys.argv[1])

print(len(test_seen_data))


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
            source = tokenizer.encode_plus(
                    self.data[index]['dialog'],
                    self.data[index]['chit-chat'],
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
            return input_ids, attn_mask, token_type_ids

    def __len__(self):
        return (len(self.data))

test_seen_set = Chit_chat_Dataset('test', test_seen_data)
test_seen_dataloader = DataLoader(test_seen_set, batch_size = 16, num_workers = 4)

model = model.to(device)
ckpt = torch.load('./model_classifer_new.ckpt')
model.load_state_dict(ckpt)
model.eval()

if fp16_training:
    model = accelerator.prepare(model) 

print("Evaluating Dev Set ...")
predict = []
with torch.no_grad():
    for i, data in enumerate(tqdm(test_seen_dataloader)):
        data[0] = data[0].to(device)
        data[1] = data[1].to(device)
        data[2] = data[2].to(device)
        outputs = model(input_ids=data[0], attention_mask=data[1], token_type_ids=data[2])

        logits = outputs.logits
        #print(logits)
    
        predict.extend(logits.cpu().numpy().tolist())
        #print('{}'.format(predict[0]))



result = []
for i, data in enumerate(test_seen_data):
    data['idx'] = i
    data['logits'] = predict[i]
    result.append(data)

with open(sys.argv[2], 'w') as f:
    json.dump(result , f, indent=2)
