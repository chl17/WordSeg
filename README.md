# Chinese Word Segmentation

## Introduction

Chinese Word Segmentation Using 

1. Maximum Matching

1. Bi-LSTM (+ CRF)

1. BERT + Bi-LSTM

## Data

SIGHAN Second International Chinese Word Segmentation Bakeoff

## Result

Trained and tested with pku dataset

### MM

|Method|Precision|Recall|F1-Score|
|:-----|:------:|:------:|:------:|
|FMM|0.802|0.781|0.791|
|RMM|0.805|0.784|0.794|
|BIMM|0.806|0.785|0.795|

### CRF

|Template|Precision|Recall|F1-Score|
|:-----|:------:|:------:|:------:|
|crf_template|0.938|0.923|0.931|

### Bi-LSTM

|Structure|Precision|Recall|F1-Score|
|:-----|:------:|:------:|:------:|
|emb256_hid256_l3|0.9343|0.9336|0.9339|

### Bi-LSTM + CRF

|Structure|Precision|Recall|F1-Score|
|:-----|:------:|:------:|:------:|
|emb256_hid256_l3|0.9343|0.9336|0.9339|

### BERT + Bi-LSTM

|Structure|Precision|Recall|F1-Score|
|:-----|:------:|:------:|:------:|
|emb768_hid512_l2|0.9698|0.9650|0.9646|

## Usages

### MM

1. select a method in `dict_based.py` and run

1.  `mm_score.bat` for scoring

### CRF

1. `crf.py` for preprocess and postprocess

1. `crf_*.bat` scripts for training, testing and scoring

### Bi-LSTM

1. Edit model configs in `models/config.py`

1. Run `python -u main.py` for training and evaluation, `python -u test.py` for evaluation only

### BERT-LSTM

1. Download pretrained models following the instruction in `pretrained_model.md`

1. Edit model config in `config.json`

1. Run `python -u train.py` for training, `python -u eval.py` for evaluation

## References

### MM

https://github.com/hiyoung123/ChineseSegmentation

### Bi-LSTM

https://github.com/luopeixiang/named_entity_recognition

### BERT-LSTM

https://github.com/AOZMH/BERT-LSTM-Chinese-Word-Segmentation
