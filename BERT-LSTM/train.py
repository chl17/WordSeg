from sklearn import metrics
import os
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel

from util import load_dataset
from model import BERT_LSTM

tag2id = {'S': 0, 'B': 1, 'M': 2, 'E': 3}
id2tag = {0: 'S', 1: 'B', 2: 'M', 3: 'E'}


def show_scores(predictions, v_labels, valid_len, f):
    score_name = ['Micro precision', 'Macro precision', 'Micro recall', 'Macro recall',
                  'Micro F1', 'Macro F1']
    scores = [0.] * 6
    for preds, golds, v_len in zip(predictions, v_labels, valid_len):
        preds = preds[1:v_len + 1]
        golds = golds[1:v_len + 1]
        scores[0] += (metrics.precision_score(preds, golds, average='micro'))
        scores[1] += (metrics.precision_score(preds, golds, average='macro'))
        scores[2] += (metrics.recall_score(preds, golds, average='micro'))
        scores[3] += (metrics.recall_score(preds, golds, average='macro'))
        scores[4] += (metrics.f1_score(preds, golds, average='micro'))
        scores[5] += (metrics.f1_score(preds, golds, average='macro'))
    for i in range(len(scores)):
        scores[i] /= len(predictions)
    for na, sc in zip(score_name, scores):
        print(na, ': ', sc)
        f.write(na + ': ' + str(sc)[:7] + '\n')
    return scores


def train():
    model_arch = train_config['model_arch']
    update_bert = train_config['update_bert']
    checkpoint_path = train_config['checkpoint_path']
    batch_size = train_config['batch_size']
    hidden_dim = lstm_config['hidden_dim']
    n_layers = lstm_config['n_layers']
    bert_dir = lstm_config['bert_dir']
    train_file = data_config['train_file']
    test_file = data_config['test_file']

    tokenizer = BertTokenizer.from_pretrained(bert_dir)

    data_loader, dev_loader = load_dataset(tokenizer, train_file, test_file, batch_size, batch_size)

    if update_bert:
        # update the embedding shape because vocab size may have changed
        bert_model = BertModel.from_pretrained('./data/transformers_pretrained')
        bert_model.resize_token_embeddings(len(tokenizer))
        bert_model.save_pretrained('./data/transformers_pretrained')
        bert_model.config.to_json_file('./data/transformers_pretrained/config.json')

    if not checkpoint_path:
        if model_arch == 'bert_lstm':
            model = BERT_LSTM(len(tag2id),
                              hidden_dim=hidden_dim,
                              bert_dir=bert_dir,
                              load_pre=True,
                              num_layers=n_layers)
    else:
        if model_arch == 'bert_lstm':
            model = BERT_LSTM(len(tag2id), hidden_dim=hidden_dim, load_pre=False, num_layers=n_layers)
        model.load_state_dict(torch.load(checkpoint_path, map_location={'cuda:1': 'cuda:0'}))

    train_bert_lstm(model, data_loader, dev_loader)


def train_bert_lstm(model, train_loader, dev_loader):
    print('Total train batch: ', len(train_loader), 'Total dev batch', len(dev_loader))
    best_macro_f1 = 0.0

    # parameters
    output_dir = train_config['output_directory']
    output_model_file = os.path.join(output_dir, 'bert_lstm.pt')
    log_output_file = os.path.join(output_dir, 'log.txt')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_config['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(train_config['epochs']):
        running_loss = 0.0

        for batch_i, (sents, tags, masks) in enumerate(train_loader):
            sents, tags, masks = sents.to(device), tags.to(device), masks.to(device)

            optimizer.zero_grad()
            out_feats = model(sents, masks)
            loss = 0.0
            # Since lossf can only calculate 2-dim output & 1-dim tag, recusively cal loss
            for b_feats, b_tags in zip(out_feats, tags):
                loss += criterion(b_feats, b_tags)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_i % 100 == 99:  # print every 100 mini-batches
                f = open(log_output_file, 'a', encoding='utf-8')
                print('[%d, %5d] loss: %.5f' %
                      (epoch + 1, batch_i + 1, running_loss / 10))
                f.write('[%d, %5d] loss: %.5f\n' %
                        (epoch + 1, batch_i + 1, running_loss / 10))
                running_loss = 0.0
                f.close()

        # evaluate every epoch
        if epoch % train_config['epochs_per_checkpoint'] == 0:
            model.eval()
            preds, golds, valid_len = [], [], []
            for sents, tags, masks in dev_loader:
                sents, tags, masks = sents.to(device), tags.to(device), masks.to(device)
                with torch.no_grad():
                    logits = model(sents, masks)

                    # predictions with regard to max score
                    preds += torch.max(logits, 2)[1].tolist()
                    # correct tags on dev set
                    golds += tags.tolist()
                    # since we must evaluate performance on raw sentence, get the length of raw sents (-2: <CLS> & <SEP>) 
                    valid_len += [int(a.sum().item() - 2) for a in masks]

            f = open(log_output_file, 'a', encoding='utf-8')
            f.write("Epoch: " + str(epoch) + '\n')
            cur_macro_f1 = show_scores(preds, golds, valid_len, f)[-1]
            print("Current best f1-macro: %s" % (str(max(cur_macro_f1, best_macro_f1))[:8]))
            f.write("Current best f1-macro: %s\n" % (str(max(cur_macro_f1, best_macro_f1))[:8]))
            f.close()

            if cur_macro_f1 > best_macro_f1:
                best_macro_f1 = cur_macro_f1
                # need to show results, thus get all input raw sentences
                all_sents = []
                for sents, _, __ in dev_loader:
                    all_sents += sents.tolist()
                # output to result file
                f = open(os.path.join(output_dir, str(cur_macro_f1)[:6] + ".dat"), "a", encoding='utf-8')
                # reload tokenizer to recover raw sentence
                tokenizer = BertTokenizer.from_pretrained(lstm_config['bert_dir'])

                for cur_preds, cur_golds, cur_len, cur_sent in zip(preds, golds, valid_len, all_sents):
                    f.write(
                        'Ques:\t' + "\t".join(tokenizer.convert_ids_to_tokens(cur_sent[1:cur_len + 1])) + '\nPred:\t' +
                        "\t".join([id2tag[temp] for temp in cur_preds[1:cur_len + 1]]) + '\nGold:\t' +
                        "\t".join([id2tag[temp] for temp in cur_golds[1:cur_len + 1]]) + '\n')
                f.close()

                print('Saving model...')
                torch.save(model.state_dict(), output_model_file)
                # model.bert_encoder.config.to_json_file(output_config_file)

            model.train()


if __name__ == '__main__':
    import json, argparse, shutil

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config.json')
    args = parser.parse_args()

    config_file = open(args.config, 'r')
    config = json.load(config_file)
    config_file.close()

    global train_config, data_config
    train_config = config['train_config']
    data_config = config['data_config']
    lstm_config = config['lstm_config']

    output_directory = train_config["output_directory"]
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    shutil.copy2(args.config, output_directory)

    train()
