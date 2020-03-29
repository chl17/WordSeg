from torchtext.data import TabularDataset, Iterator, Field
from data_preprocess import START_TAG, STOP_TAG


class Dataset:
    def __init__(self):
        self.text_field = Field(sequential=True, init_token=START_TAG, eos_token=STOP_TAG)
        self.label_field = Field(sequential=True, init_token=START_TAG, eos_token=STOP_TAG)

    def get_train_loader(self, batch_size):
        data_field = [('line', self.text_field), ('label', self.label_field)]

        train_set = TabularDataset('train_set.csv', 'csv', data_field)
        test_set = TabularDataset('test_set.csv', 'csv', data_field)
        self.text_field.build_vocab(train_set, test_set)
        self.label_field.build_vocab(train_set, test_set)

        train_loader = Iterator(train_set, batch_size)

        return train_loader

    def get_field(self):
        return self.text_field, self.label_field

    def get_vocab_size(self):
        return len(self.text_field.vocab.itos)

    def get_label_index_dict(self):
        return self.label_field.vocab.stoi


if __name__ == '__main__':
    dataset = Dataset()
    train_loader = dataset.get_train_loader(4)

    for index, batch in enumerate(train_loader):
        line = batch.line
        label = batch.label
        print(batch.line, '\n', batch.label)

