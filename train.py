# coding: UTF-8

import time
import torch
import math
import numpy as np
from train_eval import train, init_network,test
from importlib import import_module
import argparse
from dataprepare import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser()
parser.add_argument('--model',default='CNN3', type=str, help='choose a model: CNN3,RNN1,RNN2,RNN3')#, required=True
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()

import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
UNK, PAD = '<UNK>', '<PAD>'

if __name__ == '__main__':
    dataset = 'dataTest'
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'

    model_name = args.model
    embedding = 'random'

    x = import_module("."+model_name,'models')
    config = x.Config(dataset, embedding)

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)


    start_time = time.time()

    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)

    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    print("LOAD!")

    # train
    config.n_vocab = len(vocab)
    torch.cuda.current_device()

    model = x.Model(config).to(config.device)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter)
    torch.save(model, 'cnn3.pkl')
