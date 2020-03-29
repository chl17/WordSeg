import torch
import torch.nn as nn
from transformers import *
import json


class BERT_LSTM(nn.Module):

    def __init__(self, tag_size, hidden_dim, bert_dir='bert-base-chinese', load_pre=False, num_layers=1):
        super(BERT_LSTM, self).__init__()
        self.tag_size = tag_size  # num of tags for final softmax layer

        if load_pre:
            self.bert_encoder = BertModel.from_pretrained(bert_dir)
        else:
            config_file = open('./data/pretrained/config.json', 'r')
            config = json.load(config_file)
            config_file.close()

            my_config = BertConfig(**config)
            self.bert_encoder = BertModel(my_config)
        # also input dim of LSTM
        self.bert_out_dim = self.bert_encoder.config.hidden_size
        # LSTM layer
        self.lstm = nn.LSTM(self.bert_out_dim,
                            hidden_dim // 2,
                            batch_first=True,
                            num_layers=num_layers,
                            bidirectional=True)
        # map LSTM output to tag space
        self.hidden2tag = nn.Linear(hidden_dim, self.tag_size)

    def forward(self, sent, masks):
        # sent,tags,masks: (batch * seq_length)
        bert_out = self.bert_encoder(sent, masks)[0]
        # bert_out: (batch * seq_length * bert_hidden=768)
        lstm_out, _ = self.lstm(bert_out)
        # lstm_out: (batch * seq_length * lstm_hidden)
        feats = self.hidden2tag(lstm_out)
        # feats: (batch * seq_length * tag_size)

        return feats

    def eval(self):
        self.bert_encoder.eval()

    def train(self):
        self.bert_encoder.train()

