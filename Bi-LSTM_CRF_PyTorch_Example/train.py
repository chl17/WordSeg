import torch
import torch.optim as optim
from dataset import Dataset
from model import BiLSTM_CRF

# torch.set_default_tensor_type('torch.cuda.FloatTensor')

epochs = 100
dataset = Dataset()
train_loader = dataset.get_train_loader(1)
model = BiLSTM_CRF(dataset.get_vocab_size(), dataset.get_label_index_dict(), 128, 128)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

model.train()
for epoch in range(epochs):
    for iter, batch in enumerate(train_loader):
        sentence_in, targets = batch.line, batch.label

        sentence_in = sentence_in.permute([1, 0]).reshape(-1).contiguous()
        targets = targets.permute([1, 0]).reshape(-1).contiguous()

        model.zero_grad()
        loss = model.neg_log_likelihood(sentence_in.squeeze(-1), targets.squeeze(-1)) / len(sentence_in)

        loss.backward()
        optimizer.step()

        print("{}-{}: {:.5f}".format(epoch, iter, loss.item()))
