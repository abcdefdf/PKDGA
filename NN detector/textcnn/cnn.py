#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: model.py
@time:2020/12/06
@description:
"""
import random
import numpy as np
import torch
import torch.utils.data as Data
from torch import nn, optim
from sklearn.model_selection import KFold
import sklearn.metrics as metrics
from tqdm import tqdm
torch.cuda.empty_cache()

LARGE_SENTENCE_SIZE = 100  # 句子最大长度
BATCH_SIZE = 128          # 语料批次大小
LEARNING_RATE = 1e-3      # 学习率大小
EMBEDDING_SIZE = 200      # 词向量维度
KERNEL_LIST = [3, 4, 5]   # 卷积核长度
FILTER_NUM = 100          # 每种卷积核输出通道数
DROPOUT = 0.5             # dropout概率
# EPOCH = 20                # 训练轮次
EPOCH = 1                # 训练轮次
VOB_SIZE = 39

alphabet = [' ', 'g', 'o', 'l', 'e', 'y', 'u', 't', 'b', 'm', 'a', 'i', 'd', 'q', 's', 'h', 'f', 'c', 'k', '3', '6',
            '0', 'j', 'z', 'n', 'w', 'p', 'r', 'x', 'v', '1', '8', '7', '2', '9', '-', '5', '4', '_']
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
benign_txt = "C:/Users/sxy/Desktop/experiment/dataset/train/benign.txt"
malicious_txt = 'C:/Users/sxy/Desktop/experiment/dataset/train/malicious.txt'


class TextCNN(nn.Module):
    # output_size为输出类别（2个类别，0和1）,三种kernel，size分别是3,4，5，每种kernel有100个
    def __init__(self, vocab_size, embedding_dim, output_size, filter_num=100, kernel_list=(3, 4, 5), dropout=0.5):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 1表示channel_num，filter_num即输出数据通道数，卷积核大小为(kernel, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1, filter_num, (kernel, embedding_dim)),
                          nn.LeakyReLU(),
                          nn.MaxPool2d((LARGE_SENTENCE_SIZE - kernel + 1, 1)))
            for kernel in kernel_list
        ])
        self.fc = nn.Linear(filter_num * len(kernel_list), output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)  # [128, 50, 200] (batch, seq_len, embedding_dim)
        x = x.unsqueeze(1)     # [128, 1, 50, 200] 即(batch, channel_num, seq_len, embedding_dim)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)   # [128, 300, 1, 1]，各通道的数据拼接在一起
        out = out.view(x.size(0), -1)  # 展平
        out = self.dropout(out)        # 构建dropout层
        logits = self.fc(out)          # 结果输出[128, 2]
        return logits

    def predict_proba(self, test_x):
        return self.forward(test_x)


def get_data(domain, label):
    data = []
    for i in domain:
        temp = []
        for j in i.lower():
            temp.append(char_to_int[j])
        if len(temp) < LARGE_SENTENCE_SIZE:
            zeros = [0] * (LARGE_SENTENCE_SIZE - len(temp))
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


def train(model, opt, loss_function, dataloader, cuda):
    """
    训练函数
    :param model: 模型
    :param opt: 优化器
    :param loss_function: 使用的损失函数
    :return: 该轮训练模型的损失值
    """
    loss_avg = []
    model.train()  # 模型处于训练模式
    # 批次训练
    for x_batch, y_batch in tqdm(dataloader):
        x_batch = torch.LongTensor(x_batch)  # 需要是Long类型
        y_batch = torch.tensor(y_batch).long()
        # y_batch = y_batch.squeeze()  # 数据压缩到1维
        if cuda:
            x_batch = x_batch.cuda(0)
            y_batch = y_batch.cuda(0)
        pred = model(x_batch)        # 模型预测
        # 使用损失函数计算损失值，预测值要放在前
        loss = loss_function(pred, y_batch)
        loss_avg.append(loss.cpu().data)
        # 清楚之前的梯度值
        opt.zero_grad()
        # 反向传播
        loss.backward()
        # 参数更新
        opt.step()
    return np.array(loss_avg).mean()


def training(domain, label, cuda, trainnum):
    domain, label = get_data(domain, label)
    dis = TextCNN(VOB_SIZE, EMBEDDING_SIZE, 2)
    # 优化器选择
    optimizer = optim.Adam(dis.parameters(), lr=LEARNING_RATE)
    # 损失函数选择
    criterion = nn.CrossEntropyLoss()
    if cuda:
        dis = dis.cuda(0)
        criterion = criterion.cuda(0)
    print('Pretrain discriminator')
    index = [i for i in range(len(domain))]
    random.shuffle(index)
    domain = domain[index]
    label = label[index]
    for epoch in range(trainnum):
        dataloader = loader(domain, label)
        loss = train(dis, optimizer, criterion, dataloader, cuda)
        print('epoch={},loss={}'.format(epoch, loss))
    return dis


def predict_proba(dis, test_x):
    prob = dis.forward(test_x)
    return prob

# if __name__ == '__main__':
#     f = open(benign_txt, 'r')
#     benign = f.read().splitlines()
#     f.close()
#     f = open(malicious_txt, 'r')
#     malicious = f.read().splitlines()
#     f.close()
#     domain = benign + malicious
#     label = [0] * len(benign) + [1] * len(malicious)
#     domain, label = get_data(domain, label)
#     dis = TextCNN(VOB_SIZE, EMBEDDING_SIZE, 2)
#     # 优化器选择
#     optimizer = optim.Adam(dis.parameters(), lr=LEARNING_RATE)
#     # 损失函数选择
#     criterion = nn.CrossEntropyLoss()
#     dis = dis.cuda(0)
#     criterion = criterion.cuda(0)
#     print('Pretrain discriminator')
#
#     auc = 0.0
#     k = 5
#     kf = KFold(n_splits=k)
#     index = [i for i in range(len(domain))]
#     random.shuffle(index)
#     domain = domain[index]
#     label = label[index]
#     for train_index, test_index in kf.split(domain):
#         train_x, train_y = domain[train_index], label[train_index]
#         test_x, test_y = domain[test_index], label[test_index]
#         dataloader = loader(domain, label)
#         for epoch in range(EPOCH):
#             # 模型训练
#             loss = train(dis, optimizer, criterion, dataloader)
#             print('epoch={},loss={}'.format(epoch, loss))
#         with torch.no_grad():
#             test_x = test_x.cuda(0)
#             prob = dis.forward(test_x)
#             t_auc = metrics.roc_auc_score(test_y, prob[:, 1].cpu().detach().numpy())
#             auc += t_auc
#             print(t_auc)
#     print(auc)

