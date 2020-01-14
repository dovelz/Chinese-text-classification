import torch
import numpy as np
import argparse
from dataprepare import build_dataset, build_iterator
from sklearn import metrics
from importlib import import_module
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--model',default='CNN3', type=str, help='choose a model: CNN3,RNN2,RNN3')#, required=True
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()

def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    # print("Confusion Matrix...")
    # print(test_confusion)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            # print(texts)
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)

if __name__ == '__main__':
    dataset = 'dataTest'
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model
    x = import_module("." + model_name, 'models')
    config = x.Config(dataset, embedding)
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    test_iter = build_iterator(test_data, config)
    model2 = torch.load('rnn3.pkl')
    test(config, model2, test_iter)