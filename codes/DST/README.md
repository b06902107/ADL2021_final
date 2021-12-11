# Dialogue State Tracking

## Environment

Conda is required. If you haven't installed it, please refer to [documentation](https://docs.conda.io/en/latest/miniconda.html).

Then, please create the environment with the following commands:

```
cd ./env/
bash create.sh
conda activate dst
```

## Training

Three models are required to reproduce our best results in Kaggle. If you want to train them by yourself, please refer to the following commands.

### Span-DST

This model is used for non-categorical slots in both seen and unseen task.

```
cd ./NoncatSpan/
python make_data.py -d [train data dir path] [dev data dir path] -s [schema path] -o task_data/all.jsonl -l -a 0.6 --norm
python train.py --train_file task_data/all.jsonl --model_name xlnet-large-cased
```

### Choice-DST (deep)

This model is used for categorical slots in seen task.

```
cd ./CatChoice/
python make_data.py -d [train data dir path] [dev data dir path] -s [schema path] -o task_data/all.jsonl -l -a 0.6 --norm
python train.py --train_file task_data/all.jsonl --model_name roberta-large
```

### Choice-DST (deep and wide)

This model is used for categorical slots in unseen task.

```
cd ./CatChoice_WD/
python make_data.py -d [train data dir path] [dev data dir path] -s [schema path] -o task_data/all.jsonl -l -a 0.6 --norm
python train.py --train_file task_data/all.jsonl --model_name roberta-large
```


## Inference

You can reproduce the best results in Kaggle with the following commands.

First, you should download and extract our model checkpoints.
```
python download.py
```

Then, just run the commands below for reproduction. Note that the data directory must contain *test_seen/*, *test_unseen/* and *schema.json*.
```
cd ./Merge/
bash run.sh [data dir path]
```

Finally, *./Merge/outputs/test_seen/results.csv* and *./Merge/outputs/test_unseen/results.csv* are the submission files for seen task and unseen task, respectively.
