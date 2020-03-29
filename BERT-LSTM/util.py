from transformers import *
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

tag2id = {'S': 0, 'B': 1, 'M': 2, 'E': 3}
id2tag = {0: 'S', 1: 'B', 2: 'M', 3: 'E'}


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, qtok, tags):
        self.toks = qtok
        self.tags = tags

    def __len__(self):
        return len(self.toks)

    def __getitem__(self, idx):
        tok, tag = self.toks[idx], self.tags[idx]
        return tok, tag


def custom_collate(batch):
    transposed = list(zip(*batch))
    lst = []
    # transposed[0]: list of token ids of text
    padded_seq = []
    max_seq_len = len(max(transposed[0], key=len))
    for seq in transposed[0]:
        padded_seq.append(seq + [0] * (max_seq_len - len(seq)))
    lst.append(torch.LongTensor(padded_seq))

    # tansposed[1]: list of tag ids of SAME LENGTH!
    padded_tag = []
    att_mask = []
    for seq in transposed[1]:
        padded_tag.append(seq + [0] * (max_seq_len - len(seq)))
        att_mask.append([1] * len(seq) + [0] * (max_seq_len - len(seq)))
    lst.append(torch.LongTensor(padded_tag))
    lst.append(torch.FloatTensor(att_mask))

    return lst


def load_dataset(tokenizer, train_file, test_file, train_batch, dev_batch, max_sent_len=256, extra_info=False):
    f = open(train_file, 'r', encoding='utf-8')
    train_lines = f.read().strip().split('\n')
    f.close()

    f = open(test_file, 'r', encoding='utf-8')
    dev_lines = f.read().strip().split('\n')
    f.close()

    train_lines = [line.strip().split() for line in train_lines]
    dev_lines = [line.strip().split() for line in dev_lines]

    train_pruned_lines, dev_pruned_lines = [], []

    # prun data set
    for train_line in train_lines:
        current_length = 0
        line = []
        for word in train_line:
            line.append(word)
            current_length += len(word)
            if current_length > max_sent_len:
                train_pruned_lines.append(line.copy())
                current_length = 0
                line = []
        if len(line) == 0:
            continue
        train_pruned_lines.append(line.copy())

    extra_pos = []
    for dev_line in dev_lines:
        current_length = 0
        line = []
        for word in dev_line:
            line.append(word)
            current_length += len(word)
            if current_length > max_sent_len:
                dev_pruned_lines.append(line.copy())
                current_length = 0
                line = []
                extra_pos.append(len(dev_pruned_lines))
        if len(line) == 0:
            continue
        dev_pruned_lines.append(line.copy())

    # train data
    train_sents, train_tags = [], []
    for i, line in enumerate(train_pruned_lines):
        cur_sent = ''.join(line)
        pos = 0
        cur_tag = [0] * (len(cur_sent))
        if len(cur_sent) == 0:
            # print(i,line)
            continue
        for word in line:
            if len(word) == 1:
                # single word
                cur_tag[pos] = tag2id['S']
                pos += 1
            else:
                # more than one word
                cur_tag[pos] = tag2id['B']
                cur_tag[pos + len(word) - 1] = tag2id['E']
                if len(word) > 2:
                    cur_tag[pos + 1:pos + len(word) - 1] = [tag2id['M']] * (len(word) - 2)
                pos = pos + len(word)

        train_sents.append(cur_sent)
        train_tags.append([0] + cur_tag + [0])

    # dev data
    dev_sents, v_tags = [], []
    for i, line in enumerate(dev_pruned_lines):
        cur_sent = ''.join(line)
        pos = 0
        cur_tag = [0] * (len(cur_sent))
        if len(cur_sent) == 0:
            print(i, line)
            continue
        # print(line)
        for word in line:
            if len(word) == 1:
                # single word
                cur_tag[pos] = tag2id['S']
                pos += 1
            else:
                # more than one word
                cur_tag[pos] = tag2id['B']
                cur_tag[pos + len(word) - 1] = tag2id['E']
                if len(word) > 2:
                    cur_tag[pos + 1:pos + len(word) - 1] = [tag2id['M']] * (len(word) - 2)
                pos = pos + len(word)

        dev_sents.append(cur_sent)
        v_tags.append([0] + cur_tag + [0])

    # convert to token ids
    train_tokens = []
    for i, sent in tqdm(enumerate(train_sents), ncols=100, total=len(train_sents)):
        tokens = list(sent)
        token_ids = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(tokens) + [tokenizer.sep_token_id]
        train_tokens.append(token_ids)

    print("\n")
    dev_tokens = []
    for i, sent in tqdm(enumerate(dev_sents), ncols=100, total=len(dev_sents)):
        tokens = list(sent)
        token_ids = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(tokens) + [tokenizer.sep_token_id]
        dev_tokens.append(token_ids)

    dataset = MyDataset(train_tokens, train_tags)
    train_loader = DataLoader(dataset=dataset, batch_size=train_batch, collate_fn=custom_collate, shuffle=True)
    dataset = MyDataset(dev_tokens, v_tags)
    dev_loader = DataLoader(dataset=dataset, batch_size=dev_batch, collate_fn=custom_collate, shuffle=False)

    if not extra_info:
        return train_loader, dev_loader
    else:
        return train_loader, dev_loader, extra_pos


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    train_loader, dev_loader, extra_pos = load_dataset(tokenizer, 4, 4, extra_info=True)
    print(extra_pos)

    tmp = list(train_loader)
    text_batch, label_batch, masks_batch = tmp[10]
    print("\n")
    for text_ids, label_ids, mask in zip(text_batch, label_batch, masks_batch):
        text = tokenizer.convert_ids_to_tokens(text_ids.tolist())[:int(mask.sum())]
        print(len(text), len(label_ids))
        for word in text:
            print(word + '\t', end='')
        print("\n")
        for label_id in label_ids:
            print(id2tag[label_id.item()] + '\t', end='')
        print("\n")
