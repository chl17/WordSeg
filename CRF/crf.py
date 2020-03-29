import codecs


def tag_train(input_file, output_file):
    input_data = codecs.open(input_file, 'r', 'utf-8')
    output_data = codecs.open(output_file, 'w', 'utf-8')
    for line in input_data.readlines():
        word_list = line.strip().split()
        for word in word_list:
            if len(word) == 1:
                output_data.write(word + "\tS\n")
            else:
                output_data.write(word[0] + "\tB\n")
                for w in word[1:len(word) - 1]:
                    output_data.write(w + "\tM\n")
                output_data.write(word[len(word) - 1] + "\tE\n")
        output_data.write("\n")
    input_data.close()
    output_data.close()


def split_test(input_file, output_file):
    input_data = codecs.open(input_file, 'r', 'utf-8')
    output_data = codecs.open(output_file, 'w', 'utf-8')
    for line in input_data.readlines():
        for word in line.strip():
            word = word.strip()
            if word:
                output_data.write(word + "\tB\n")
        output_data.write("\n")
    input_data.close()
    output_data.close()


def cat_test(input_file, output_file):
    input_data = codecs.open(input_file, 'r', 'utf-8')
    output_data = codecs.open(output_file, 'w', 'utf-8')
    for line in input_data.readlines():
        if line == "\n" or line == "\r\n":
            output_data.write("\n")
        else:
            char_tag_pair = line.strip().split('\t')
            char = char_tag_pair[0]
            tag = char_tag_pair[2]
            if tag == 'B':
                output_data.write(' ' + char)
            elif tag == 'M':
                output_data.write(char)
            elif tag == 'E':
                output_data.write(char + ' ')
            else:  # tag == 'S'
                output_data.write(' ' + char + ' ')
    input_data.close()
    output_data.close()


# remove '\r\n' line in train set
def process_train():
    with codecs.open('../Data/train/pku_training.utf8', 'r', 'utf-8') as original:
        with codecs.open('../Data/train/pku_training_processed.utf8', 'w', 'utf-8') as processed:
            processed.writelines([line for line in original if line != '\r\n'])


if __name__ == '__main__':
    # remove '\r\n' lines in train set
    # process_train()

    # add B E M S label
    # tag_train('../Data/train/pku_training_processed.utf8', 'pku_training_tagged.txt')

    # crf train
    # crf_train.bat

    # split test set into lines with format "word B", ready for crf test, B is placeholder
    # split_test('../Data/test/pku_test.utf8', 'pku_test_split.txt')
    # split_test('../Data/train/pku_training.utf8', 'pku_training_split.txt')

    # crf test
    # crf_test.bat

    # cat output of crf test "word B label" to space split format
    cat_test('pku_test_tagged.txt', 'pku_test_result.txt')
    cat_test('pku_training_test_tagged.txt', 'pku_training_result.txt')

    # crf score
    # crf_score.bat
