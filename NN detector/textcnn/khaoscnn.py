import os
import sys
import time
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from tqdm import tqdm
import torch.utils.data as Data
import torch.optim as optim
import khaos.language_helpers as language_helpers
import khaos.tflib as lib
from sklearn.preprocessing import OneHotEncoder
import khaos.N_gram as N_gram

sys.path.append(os.getcwd())
torch.manual_seed(1)
# use_cuda = torch.cuda.is_available()
use_cuda = True
if use_cuda:
    gpu = 0

# Download Google Billion Word at http://www.statmt.org/lm-benchmark/ and
# fill in the path to the extracted files here!
DATA_DIR = 'benign.txt'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_language.py!')
LARGE_SENTENCE_SIZE = 100  # 句子最大长度
LEARNING_RATE = 1e-3      # 学习率大小
BATCH_SIZE = 64  # Batch size
ITERS = 55  # How many iterations to train for
SEQ_LEN = 100  # Sequence length in characters
DIM = 64  # Model dimensionality. This is fairly slow and overfits, even on
          # Billion Word. Consider decreasing for smaller datasets.
CRITIC_ITERS = 156 # How many critic iterations per generator iteration. We
                  # use 10 for the results in the paper, but 5 should work fine
                  # as well.
LAMBDA = 10  # Gradient penalty lambda hyperparameter.
MAX_N_EXAMPLES = 10000  # 10000000 # Max number of data examples to load. If data loading
                          # is too slow or takes too much RAM, you can decrease
                          # this (at the expense of having less training data).
alphabet = [' ', 'g', 'o', 'l', 'e', 'y', 'u', 't', 'b', 'm', 'a', 'i', 'd', 'q', 's', 'h', 'f', 'c', 'k', '3', '6',
            '0', 'j', 'z', 'n', 'w', 'p', 'r', 'x', 'v', '1', '8', '7', '2', '9', '-', '5', '4', '_']
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
benign_txt = "C:/Users/sxy/Desktop/experiment/dataset/train/benign.txt"
malicious_txt = 'C:/Users/sxy/Desktop/experiment/dataset/train/malicious.txt'


class ResBlock(nn.Module):

    def __init__(self):
        super(ResBlock, self).__init__()

        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(DIM, DIM, 3, padding=1),  # nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Conv1d(DIM, DIM, 3, padding=1),  # nn.Linear(DIM, DIM),
        )

    def forward(self, input):
        output = self.res_block(input)
        return input + (0.3*output)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1d = nn.Conv1d(39, DIM, 1)
        self.block = nn.Sequential(
            ResBlock(),
            ResBlock(),
            ResBlock(),
        )
        self.linear = nn.Linear(SEQ_LEN*DIM, 2)

    def forward(self, input):
        output = input.transpose(1, 2)  # (BATCH_SIZE, len(charmap), SEQ_LEN)
        output = self.conv1d(output)
        output = self.block(output)
        output = output.view(-1, SEQ_LEN*DIM)
        output = self.linear(output)
        return output


def get_data(domain, label):
    data1 = []
    for i in domain:
        temp = []
        for j in i.lower():
            temp.append(char_to_int[j])
        if len(temp) < LARGE_SENTENCE_SIZE:
            zeros = [0] * (LARGE_SENTENCE_SIZE - len(temp))
            temp = zeros + temp
        data1.append(temp)
    data = []
    s = (100, 39)
    # label = torch.LongTensor(integer_encoded)
    for charint in data1:
        one_hot1 = np.zeros(s, dtype=torch.LongTensor)
        for (j, k) in zip(one_hot1, charint):
            if k == 0:
                continue
            j[k] = 1
        # one_hot1 = one_hot1.T
        if len(one_hot1) != 100:
            print('len', len(one_hot1))
        data.append(one_hot1)  # one_hot1 is <class 'numpy.ndarray'>
    # data = np.array(data)
    # label = np.array(label)
    data = torch.Tensor(data)
    label = torch.Tensor(label)
    return data, label
    # return one_hot, label


def loader(data, label):
    dataset = Data.TensorDataset(data, label)
    # shuffle 是打乱数据， drop_last是总样本数不能整除batch size
    loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    return loader


def train(model, opt, loss_function, dataloader):
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
        # x_batch = x_batch.clone().detach().long()
        # x_batch = torch.tensor(x_batch)  # 需要是Long类型
        x_batch = Variable(x_batch)
        y_batch = torch.tensor(y_batch).long()
        # y_batch = y_batch.squeeze()  # 数据压缩到1维
        # x_batch = x_batch.cuda(0)
        # y_batch = y_batch.cuda(0)
        # print(x_batch)
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


def training(domain, label):
    domain, label = get_data(domain, label)
    dis = Discriminator()
    # 优化器选择
    optimizer = optim.Adam(dis.parameters(), lr=LEARNING_RATE)
    # 损失函数选择
    criterion = nn.CrossEntropyLoss()
    # dis = dis.cuda(0)
    # criterion = criterion.cuda(0)
    print('Pretrain discriminator')

    for epoch in range(1):
        dataloader = loader(domain, label)
        loss = train(dis, optimizer, criterion, dataloader)
        print('loss', loss)
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

