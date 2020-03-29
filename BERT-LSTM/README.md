# BERT-LSTM-Chinese-Word-Segmentation
BERT-LSTM-based Chinese word segmentation model on SIGHAN-2004

## Data Prepare
Please use /data directory at https://disk.pku.edu.cn:443/link/E65BE6291BB4D4841F29ACD28F51FC81 to replace the blank directory here.

Such file contains the pretrained parameters and all fine-tuned results. A PKU-net-disk account is required :).

## Requirements
Requires sklearn, pytorch, transformers package.

## Results
Tested on SIGHAN-2004 Chinese Word Segmentation dataset

|Measurements|Performance|
|:--------------|:----------:|
|TOTAL INSERTIONS|593|
|TOTAL DELETIONS|639|
|TOTAL SUBSTITUTIONS|1053|
|TOTAL NCHANGE|2285|
|OOV Rate|0.026|
|OOV Recall Rate|0.854|
|IV Recall Rate|0.988|
|TOTAL TRUE WORD COUNT|	106873|
|TOTAL TEST WORD COUNT|106827|
|**TOTAL TRUE WORDS RECALL**|**0.984**|
|**TOTAL TEST WORDS PRECISION**|**0.985**|
|**F MEASURE**|**0.984**|

## Execution
Train model:
> python main.py

Currently only support BERT-LSTM model.

As shown in the uncompleted functions in main.py, we are working on other model architectures for Chinese-seg, e.g. LSTM-CRF, BERT-LSTM-CRF, the results will be shown later.

Evluate:
> python eval.py

This execution will create a file (e.g. one that named 'test_pred_bert_lstm_1.txt') in /eval directory, which contains the segmentated results on test data at /data/test.txt. 

Currently we haven't incorporate argument parsers in the codes above, so please mannually change the corresponding details in the code to assign the names and routes of the files including logs, results and checkpoints.

## Performance comparison
The comparison between our results and the open-source tool Pkuseg.

|Model|OOV Rate|OOV Recall|IV Recall|Recall|Precision|F Measure|
|:--------------|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
|Pkuseg|0.026|**0.873**|0.413|0.883|0.863|0.873|
|Our Model|0.026|0.854|**0.988**|**0.984**|**0.985**|**0.984**|
