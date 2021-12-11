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

random.seed(811)
np.random.seed(811)
torch.manual_seed(811)
torch.cuda.manual_seed(811)
torch.cuda.manual_seed_all(811)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

fp16_training = False

if fp16_training:
    #!pip install accelerate==0.2.0
    from accelerate import Accelerator
    accelerator = Accelerator(fp16=True)
    device = accelerator.device 


def read_data(data_dir):
    data = []
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        with open(file_path, 'r', encoding="utf-8") as reader:
            data.extend(json.load(reader))
    return data

def data_select_test(data):
    new_data = []
    for dialogue in data:
        #temp_dict = {'dialogue_id' : dialogue['dialogue_id']}
        one_data = 'summarize: '
        for turn in dialogue['turns']:
            if turn['speaker'] == 'USER':
                one_data += 'USER : ' + turn['utterance'] + ' '
            elif turn['speaker'] == 'SYSTEM':
                #print(turn['frames'])
                #temp_dict['dialog'] = "".join(one_data)
                #new_data.append(temp_dict)
                new_data.append({'dialogue_id' : dialogue['dialogue_id'], 'turn_id' : turn['turn_id'], 'location' : 'begin', 'dialog' : "".join(one_data)})

                one_data += 'SYSTEM : ' + turn['utterance'] + ' '

                #temp_dict2 = temp_dict.copy()
                #temp_dict2['dialog'] = "".join(one_data)
                #new_data.append(temp_dict2)
                new_data.append({'dialogue_id' : dialogue['dialogue_id'], 'turn_id' : turn['turn_id'], 'location' : 'end', 'dialog' : "".join(one_data)})

    return new_data

#test_unseen_data = data_select_test(read_data('./data-0614/test_unseen'))
data_dir = sys.argv[1]
test_seen_data = data_select_test(read_data(os.path.join(data_dir, 'test_seen')))

print(len(test_seen_data))
#print(len(test_unseen_data))
#print(test_unseen_data[0:16])

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
            return input_ids, attn_mask, self.data[index]['dialogue_id']

    def __len__(self):
        return (len(self.data))


test_seen_set = Chit_chat_Dataset('test', test_seen_data)
#test_unseen_set = Chit_chat_Dataset('test', test_unseen_data)

test_seen_dataloader = DataLoader(test_seen_set, batch_size = 16, num_workers = 4)
#test_unseen_dataloader = DataLoader(test_unseen_set, batch_size = 16, num_workers = 4)

model = model.to(device)

def ids_to_clean_text(generated_ids):
    gen_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return list(map(str.strip, gen_text))


print("Evaluating Dev Set ...")
ckpt = torch.load('./model_NLG.ckpt')
model.load_state_dict(ckpt)
model.eval()
predict = []
with torch.no_grad():
    for i, data in enumerate(tqdm(test_seen_dataloader)):
        data[0] = data[0].to(device)
        data[1] = data[1].to(device)
        #labels = data[2]
        #outputs = model(input_ids=data[0], attention_mask=data[1], decoder_input_ids=data[2], decoder_attention_mask=data[3], labels = labels)
        #output = model.generate(input_ids=data[0], attention_mask=data[1], max_length=150, early_stopping=True, use_cache=True, 
        #    repetition_penalty=2.5, length_penalty=1.0)
        #output = model.generate(input_ids=data[0], attention_mask=data[1], max_length=50, early_stopping=True, use_cache=True, 
        #    repetition_penalty=2.5, length_penalty=1.0, do_sample=False, num_beams=5)

        output = model.generate(input_ids=data[0], attention_mask=data[1], max_length=50, early_stopping=True, use_cache=True, 
            repetition_penalty=2.5, length_penalty=1.0, do_sample=True, num_beams=5, top_k = 640)

        output = ids_to_clean_text(output)
        #target = ids_to_clean_text(data[2])
        #preds[line['id']] = line['title'].strip() + '\n'
        #refs[line['id']] = line['title'].strip() + '\n'
        #val_loss += outputs.loss.item()
        #print(output)
        '''
        new_output = []
        for i, j in enumerate(output):
            chin = False
            for c in j:
                if c >= u'\u4e00' and c <= u'\u9fa5':
                    chin = True
                if c.isalpha() == True:
                    chin = True
            if chin == False:
                print(j)
                print(target[i])
                j = j + '空'
                print(len(j))
            new_output.append(j)
        '''
    
        predict.extend(output)
        #targets.extend(target)

    result = []
    for i, data in enumerate(test_seen_data):
        result.append({'id': data['dialogue_id'], 'turn_id' : data['turn_id'], 'location' : data['location'], 'dialog': data['dialog'], 'chit-chat' : predict[i]})

    with open(sys.argv[2], 'w') as f:
        json.dump(result , f, indent=2)

'''
predict = []
with torch.no_grad():
    for i, data in enumerate(tqdm(test_unseen_dataloader)):
        data[0] = data[0].to(device)
        data[1] = data[1].to(device)
        #labels = data[2]
        #outputs = model(input_ids=data[0], attention_mask=data[1], decoder_input_ids=data[2], decoder_attention_mask=data[3], labels = labels)
        #output = model.generate(input_ids=data[0], attention_mask=data[1], max_length=150, early_stopping=True, use_cache=True, 
        #    repetition_penalty=2.5, length_penalty=1.0)
        output = model.generate(input_ids=data[0], attention_mask=data[1], max_length=50, early_stopping=True, use_cache=True, 
            repetition_penalty=2.5, length_penalty=1.0, do_sample=False, num_beams=5)

        output = ids_to_clean_text(output)  
        predict.extend(output)

    result = []
    for i, data in enumerate(test_unseen_data):
        result.append({'id': data['dialogue_id'], 'dialog': data['dialog'], 'chit-chat' : predict[i]})

    with open('./ori_chit_chat_unseen_sampling.json', 'w') as f:
        json.dump(result , f)
'''