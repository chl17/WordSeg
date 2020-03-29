import pandas
import torch

START_TAG = "<START>"
STOP_TAG = "<STOP>"
tag_to_ix = {"B": 0, "M": 1, "E": 2, "S": 3, START_TAG: 4, STOP_TAG: 5}


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# convert line to chars
def get_char(sentence):
    return list(''.join(sentence.split(' ')))


# convert line to B E M S label
def get_label(sentence):
    output_str = []
    word_list = sentence.split()
    for i in range(len(word_list)):
        if len(word_list[i]) == 1:
            output_str.append('S')
        elif len(word_list[i]) == 2:
            output_str.append('B')
            output_str.append('E')
        else:
            M_num = len(word_list[i]) - 2
            output_str.append('B')
            output_str.extend('M' * M_num)
            output_str.append('E')
    return output_str


def read_file(filename):
    char_list, line_list, label_list, label_line_list = [], [], [], []

    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        line_char_list = get_char(line)
        line_label_list = get_label(line)

        char_list.extend(line_char_list)
        line_list.append(line_char_list)
        label_list.extend(line_label_list)
        label_line_list.append(line_label_list)
    return char_list, line_list, label_list, label_line_list


if __name__ == '__main__':
    char_list, line_list, label_list, label_line_list = read_file('../Data/train/pku_training.utf8')
    char_list = [' '.join(line) for line in line_list]
    label_list = [' '.join(line) for line in label_line_list]
    data = pandas.DataFrame({'line': char_list, 'label': label_list})
    data.to_csv('train_set.csv', encoding='utf-8', index=False, header=False)

    char_list, line_list, label_list, label_line_list = read_file('../Data/gold/pku_test_gold.utf8')
    char_list = [' '.join(line) for line in line_list]
    label_list = [' '.join(line) for line in label_line_list]
    data = pandas.DataFrame({'line': char_list, 'label': label_list})
    data.to_csv('test_set.csv', encoding='utf-8', index=False, header=False)
