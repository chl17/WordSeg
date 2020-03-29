import codecs
from tqdm import tqdm


class FMM(object):
    def __init__(self, word_dict):
        self.word_dict = word_dict
        self.window_size = self.get_max_length()

    def get_max_length(self):
        # get max length of word in dict
        return max(map(len, [w for w in self.word_dict]))

    def cut(self, text):
        result = []
        index = 0
        text_size = len(text)
        while text_size > index:
            for size in range(self.window_size + index, index, -1):
                piece = text[index:size]
                if piece in self.word_dict:
                    index = size - 1
                    break
            index = index + 1
            result.append(piece)
        return result


class RMM(object):
    def __init__(self, word_dict):
        self.word_dict = word_dict
        self.window_size = self.get_max_length()

    def get_max_length(self):
        # get max length of word in dict
        return max(map(len, [w for w in self.word_dict]))

    def cut(self, text):
        result = []
        index = len(text)
        window_size = min(index, self.window_size)
        while index > 0:
            for size in range(index - window_size, index):
                piece = text[size:index]
                if piece in self.word_dict:
                    index = size + 1
                    break
            index = index - 1
            result.append(piece)
        result.reverse()
        return result


class BIMM(object):
    def __init__(self, word_dict):
        self.word_dict = word_dict
        self.FMM = FMM(self.word_dict)
        self.RMM = RMM(self.word_dict)

    def cut(self, text):
        res_fmm = self.FMM.cut(text)
        res_rmm = self.RMM.cut(text)
        if len(res_fmm) == len(res_rmm):
            if res_fmm == res_rmm:
                return res_fmm
            else:
                f_word_count = len([w for w in res_fmm if len(w) == 1])
                r_word_count = len([w for w in res_rmm if len(w) == 1])
                return res_fmm if f_word_count < r_word_count else res_rmm
        else:
            return res_fmm if len(res_fmm) < len(res_rmm) else res_rmm


def read_dict(dic_path):
    dic = []
    with open(dic_path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().split()
            dic.append(word[0])
    return dic


def run_test(tokenizer, input_path, output_path, lines=-1):
    with codecs.open(input_path, 'r', 'utf-8') as file:
        input_lines = file.readlines()

    if lines != -1:
        input_lines = input_lines[:lines]
        with codecs.open(output_path + '.test.txt', 'w', 'utf-8') as file:
            file.writelines(input_lines)

    output_lines = []
    for line in tqdm(input_lines, ncols=100):
        word_list = tokenizer.cut(line)
        output_lines.append(' '.join(word_list))

    with codecs.open(output_path, 'w', 'utf-8') as file:
        file.writelines(output_lines)


if __name__ == '__main__':
    dic = read_dict("../Data/dict.txt")

    # tokenizer = FMM(dic)
    tokenizer = RMM(dic)
    # tokenizer = BIMM(dic)
    input_text = "../Data/test/pku_test.utf8"
    output_text = "../MM/RMM.txt"

    run_test(tokenizer, input_text, output_text, -1)
