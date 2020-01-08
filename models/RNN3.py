# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):

    def __init__(self, dataset, embedding):
        self.model_name = 'RNN3'
        self.train_path = dataset + '/data/train.txt'
        self.dev_path = dataset + '/data/dev.txt'
        self.test_path = dataset + '/data/test.txt'
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]
        self.vocab_path = dataset + '/data/vocab.pkl'
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'

        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dropout = 0.9
        self.require_improvement = 1000
        self.num_classes = len(self.class_list)
        self.n_vocab = 0
        self.num_epochs = 20
        self.batch_size = 128
        self.pad_size = 50
        self.learning_rate = 1e-3
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300
        self.hidden_size = 128
        self.num_layers = 2




class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(298, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.gru =nn.GRU(2*config.hidden_size,config.hidden_size, config.num_layers,
                                   batch_first=True,bidirectional=True,dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)
        self.m = nn.Softmax(dim=1)
        self.loss_fn =nn.MSELoss(reduction='sum')
        self.conv=nn.Conv2d(1, 1, (3, 3))

    def forward(self, x):
        x, _ = x
        embed = self.embedding(x)
        out = self.embedding(x)
        out = out.unsqueeze(1)
        out =self.conv(out)#torch.Size([128, 1, 24, 292])
        out = out.squeeze(1)
        out, hid = self.lstm(out)#torch.Size([128, 32, 256])
        out, hid =self.gru(out)#([128, 32, 256])
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        out = self.m(out)#128,60
        return out

  