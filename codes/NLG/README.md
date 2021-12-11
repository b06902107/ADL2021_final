# Natural Language Generation

## Requirements
transformers
numpy
torch
tqdm
gdown
json

## Train

### Training T5 model
```bash
python3 ./NLG.py data_dir
```

where data_dir is the folder contianing train and dev data folder 

### Training RoBERTa model
```bash
python3 ./train_classifier.py data_dir
```

where data_dir is the folder contianing train and dev data folder

## Generate test chit-chat response

```bash
#download model first
python3 ./download.py
python3 ./gen_ori_chit_chat.py data_dir ori_chit_chat_result_path
python3 ./select_chit_chat.py ori_chit_chat_result_path ori_chit_chat_result_add_logits_path
python3 ./select_chit_chat_with_logits.py ori_chit_chat_result_add_logits_path final_result_path
```

where data_dir is test seen data folder, 
ori_chit_chat_result_path is original chit-chat result file, 
ori_chit_chat_result_add_logits_path is original chit-chat result adding logits file, 
final_result_path is final NLG result file, 
