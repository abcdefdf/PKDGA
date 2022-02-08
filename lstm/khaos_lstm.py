# -*- coding: utf-8 -*-

import math
import random
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
import torch.autograd as autograd
import torch
from tqdm import tqdm
from sklearn.model_selection import KFold
import sklearn.metrics as metrics

alphabet = [' ', 'g', 'o', 'l', 'e', 'y', 'u', 't', 'b', 'm', 'a', 'i', 'd', 'q', 's', 'h', 'f', 'c', 'k', '3', '6',
            '0', 'j', 'z', 'n', 'w', 'p', 'r', 'x', 'v', '1', '8', '7', '2', '9', '-', '5', '4', '_']
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
POSITIVE_FILE = 'real1.data'
NEGATIVE_FILE = './malicious.data'

benign_txt = "C:/Users/sxy/Desktop/experiment/dataset/train/benign.txt"
malicious_txt = 'C:/Users/sxy/Desktop/experiment/dataset/train/malicious.txt'
MAX_SEQ = 100
BATCH_SIZE = 64
VOCAB_SIZE = 39
TRAIN_NUM = 128926
cuda = False


class Discriminator(nn.Module):
    def __init__(self, num_class, vocab_size, emb_dim, hidden_dim, use_cuda, num_layers):
        super(Discriminator, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_cuda = use_cuda

        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        # self.lstm = nn.LSTM(vocab_size, hidden_dim, num_layers=num_layers, batch_first=True)
        self.lin = nn.Linear(hidden_dim, num_class)

        self.init_params()

    def init_hidden(self, batch_size):
        h = autograd.Variable(torch.zeros((self.num_layers, batch_size, self.hidden_dim)))
        c = autograd.Variable(torch.zeros((self.num_layers, batch_size, self.hidden_dim)))
        if self.use_cuda:
            h, c = h.cuda(), c.cuda()
        return h, c

    # def forward(self, x, data_len):
    def forward(self, x):
        self.lstm.flatten_parameters()
        emb = self.emb(x)
        h0, c0 = self.init_hidden(x.size(0))
        output, (h, c) = self.lstm(emb, (h0, c0))
        # output, (h, c) = self.lstm(x, (h0, c0))
        output = output[:, -1, :]
        output = self.lin(output.contiguous().view(-1, self.hidden_dim))
        pred = F.softmax(output, dim=1)
        return pred

    def init_params(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)

    def predict_proba(self, test_x):
        return self.forward(test_x)

    def batchClassify(self, inp):
        """
        Classifies a batch of sequences.

        Inputs: inp
            - inp: batch_size x seq_len

        Returns: out
            - out: batch_size ([0,1] score)
        """
        out = self.forward(inp)
        return out.view(-1)


def get_data(domain, label):
    data = []
    for i in domain:
        temp = []
        for j in i.lower():
            temp.append(char_to_int[j])
        if len(temp) < MAX_SEQ:
            zeros = [0] * (MAX_SEQ - len(temp))
            temp = zeros + temp
        data.append(temp)
    data = torch.tensor(data)
    label = torch.tensor(label)
    return data, label


def loader(data, label):
    dataset = Data.TensorDataset(data, label)
    # shuffle 是打乱数据， drop_last是总样本数不能整除batch size
    loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    return loader


def train_dis_epoch(model, data_iter, criterion, optimizer):
    total_loss = 0.
    num = 1
    for (data, target) in tqdm(data_iter):
        data = Variable(data)
        target = Variable(target)
        if cuda:
            data, target = data.cuda(), target.cuda()
        target = target.contiguous().view(-1)
        pred = model.forward(data)
        pred = torch.log(pred)
        loss = criterion(pred, target)
        total_loss += loss.data/target.size(0)
        # total_words += data.size(0) * data.size(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        num += 1
    return math.exp(total_loss / num)


def training(domain, label):
    domain, label = get_data(domain, label)
    dis = Discriminator(2, VOCAB_SIZE, 32, 32, cuda, 1)
    dis_criterion = nn.NLLLoss(reduction='sum')
    dis_optemizer = optim.Adam(dis.parameters())
    if cuda:
        dis = dis.cuda()
        dis_criterion = dis_criterion.cuda()
    print('Pretrain discriminator')
    index = [i for i in range(len(domain))]
    random.shuffle(index)
    domain = domain[index]
    label = label[index]
    for epoch in range(15):
    # for epoch in range(1):
        dis_data_iter = loader(domain, label)
        loss = train_dis_epoch(dis, dis_data_iter, dis_criterion, dis_optemizer)
        print('Epoch [%d], loss: %f' % (epoch, loss))
    return dis


def predict_proba(dis, test_x):
    prob = dis.forward(test_x)
    return prob


if __name__ == '__main__':
    f = open(benign_txt, 'r')
    benign = f.read().splitlines()
    f.close()
    f = open(malicious_txt, 'r')
    malicious = f.read().splitlines()
    f.close()
    domain = benign + malicious
    label = [0] * len(benign) + [1] * len(malicious)
    domain, label = get_data(domain, label)
    dis = Discriminator(2, VOCAB_SIZE, 32, 32, cuda, 1)
    dis_criterion = nn.NLLLoss(reduction='sum')
    dis_optemizer = optim.Adam(dis.parameters())
    if cuda:
        dis = dis.cuda()
        dis_criterion = dis_criterion.cuda()
    print('Pretrain discriminator')
    print(label)

    auc = 0.0
    k = 5
    kf = KFold(n_splits=k)
    index = [i for i in range(len(domain))]
    random.shuffle(index)
    domain = domain[index]
    label = label[index]
    for train_index, test_index in kf.split(domain):
        train_x, train_y = domain[train_index], label[train_index]
        test_x, test_y = domain[test_index], label[test_index]
        for epoch in range(15):
            dis_data_iter = loader(train_x, train_y)
            loss = train_dis_epoch(dis, dis_data_iter, dis_criterion, dis_optemizer)
            print('Epoch [%d], loss: %f' % (epoch, loss))
        prob = dis.forward(test_x)
        t_auc = metrics.roc_auc_score(test_y, prob[:, 1].detach().numpy())
        auc += t_auc
    print(auc)
