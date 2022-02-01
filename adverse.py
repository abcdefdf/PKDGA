# -*- coding:utf-8 -*-

import exp1
import random
import math
import sklearn.metrics as metrics
from tensorflow.keras.preprocessing import sequence
import argparse
from sklearn import tree
from collections import Counter
import numpy as np
from tkinter import _flatten
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from generate_sample.generator import Generator
import file_check
from generate_sample.target_lstm import TargetLSTM
from generate_sample.rollout import Rollout
from generate_sample.data_iter import GenDataIter
# import graph.WordGraph as wordgraph
import lstm.lstm_class as lstm
import textcnn.cnn as cnn
# ================== Parameter Definition =================

parser = argparse.ArgumentParser(description='Training Parameter')
parser.add_argument('--cuda', action='store', default=None, type=int)
opt = parser.parse_args()
print(opt)

# Basic Training Paramters
SEED = 88
BATCH_SIZE = 64
GENERATED_NUM = 9984
POSITIVE_FILE = 'real.data'
NEGATIVE_FILE = 'gene.data'
EVAL_FILE = 'eval.data'
VOCAB_SIZE = 38
TOTAL_BATCH = 60
# TOTAL_BATCH = 2
PRE_EPOCH_NUM = 120
# PRE_EPOCH_NUM = 2
reward_num = 20
# reward_num = 2
if opt.cuda is not None and opt.cuda >= 0:
    torch.cuda.set_device(opt.cuda)
    opt.cuda = True

# Genrator Parameters
g_emb_dim = 32
g_hidden_dim = 64
g_sequence_len = 100

Train_RF_Samples = 10000
benign_txt = 'C:/Users/sxy/Desktop/experiment/dataset/train/benign.txt'
malicious_txt = 'C:/Users/sxy/Desktop/experiment/dataset/train/malicious.txt'

alphabet = [' ', 'g', 'o', 'l', 'e', 'y', 'u', 't', 'b', 'm', 'a', 'i', 'd', 'q', 's', 'h', 'f', 'c', 'k', '3', '6',
            '0', 'j', 'z', 'n', 'w', 'p', 'r', 'x', 'v', '1', '8', '7', '2', '9', '-', '5', '4', '_']

int_to_char = dict((i, c) for i, c in enumerate(alphabet))
char_to_int = dict((c, i) for i, c in enumerate(alphabet))


def generate_samples(model, batch_size, generated_num, output_file):
    samples = []
    for _ in range(int(generated_num / batch_size)):
        sample = model.sample(batch_size, g_sequence_len).cpu().data.numpy().tolist()
        samples.extend(sample)
    with open(output_file, 'w') as fout:
        for sample in samples:
            string = ' '.join([str(s) for s in sample])
            fout.write('%s\n' % string)


def train_epoch(model, data_iter, criterion, optimizer):
    total_loss = 0.
    total_words = 0.
    for (data, target) in data_iter:#tqdm(
        #data_iter, mininterval=2, desc=' - Training', leave=False):
        data = Variable(data)
        target = Variable(target)
        if opt.cuda:
            data, target = data.cuda(), target.cuda()
        target = target.contiguous().view(-1)
        pred = model.forward(data)
        loss = criterion(pred, target)
        total_loss += loss.data
        total_words += data.size(0) * data.size(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    data_iter.reset()
    return math.exp(total_loss / total_words)


class GANLoss(nn.Module):
    """Reward-Refined NLLLoss Function for adversarial training of Generator"""
    def __init__(self):
        super(GANLoss, self).__init__()

    def forward(self, prob, target, reward):
        """
        Args:
            prob: (N, C), torch Variable
            target : (N, ), torch Variable
            reward : (N, ), torch Variable
        """
        print("GANLoss")
        print(reward)
        N = target.size(0)
        C = prob.size(1)
        one_hot = torch.zeros((N, C))
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        one_hot.scatter_(1, target.data.view((-1, 1)), 1)
        one_hot = one_hot.type(torch.ByteTensor)
        one_hot = Variable(one_hot)
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        loss = torch.masked_select(prob, one_hot)
        print(loss.size(), reward.size())
        loss = loss * reward
        loss = -torch.sum(loss)
        return loss


def real_samples():
    f = open(benign_txt, 'r')
    data = f.read().splitlines()
    f.close()
    data = data[:9984]
    one_hot = []
    for single_data in data:
        # integer encode input data
        integer_encoded = [char_to_int[char] for char in single_data]
        i = len(integer_encoded)
        while i < g_sequence_len:
            integer_encoded.append(0)
            i += 1
        one_hot.append(integer_encoded)
    with open('real.data', 'w') as fout:
        for sample in one_hot:
            string = ' '.join([str(s) for s in sample])
            fout.write('%s\n' % string)


def AUC(gen, dis, name, generated_num):
    samples = []
    for _ in range(int(generated_num / BATCH_SIZE)):
        sample = gen.sample(BATCH_SIZE, g_sequence_len).cpu().data.numpy().tolist()
        samples.extend(sample)
    new_str = []
    for s in samples:
        new_int = []
        for b in s:
            if (int_to_char[b] == ' '):
                continue
            new_int.append(int_to_char[b])
        str1 = ''.join(new_int)
        new_str.append(str1)
    benign = []
    benign_b = file_check.get_no_dot(benign_txt)
    resultlist = random.sample(range(0, len(benign_b)), generated_num)
    for i in resultlist:
        benign.append(benign_b[i])

    y_true = [0] * len(benign) + [1] * len(new_str)
    data = benign + new_str
    if name is 'RF':
        t_prob = dis.predict_proba(data)
        t_auc = metrics.roc_auc_score(y_true, t_prob[:, 1])
        dis_loss = ''
        gen_loss = ''
    elif name is 'Statics' or name is 'Graph':
        if name is 'Statics':
            t_probs = dis.predict_proba(data, 10000)
        else:
            t_probs = dis.predict_proba(data)
        t_auc = metrics.roc_auc_score(y_true, t_probs)
        nll_prob_dis = []
        for i in t_probs[:generated_num]:
            if i == 1:
                nll_prob_dis.append(1e-04)
            else:
                nll_prob_dis.append(1-i)
        for i in t_probs[generated_num:]:
            if i == 0:
                nll_prob_dis.append(1e-04)
            else:
                nll_prob_dis.append(1)
        # print(nll_prob_dis)
        # print(np.log(nll_prob_dis))
        dis_loss = np.average(np.log(nll_prob_dis)) * -1
        nll_prob_gen1 = t_probs[-generated_num:]
        nll_prob_gen = []
        for i in nll_prob_gen1:
            if i == 1:
                nll_prob_gen.append(1e-04)
            else:
                nll_prob_gen.append(1-i)
        gen_loss = np.average(np.log(nll_prob_gen)) * -1
    else:
        X = [[char_to_int[y] for y in x] for x in data]
        X = sequence.pad_sequences(X, 100)
        X = torch.LongTensor(X)
        t_probs = dis.predict_proba(X)
        nll_prob_dis = []
        nll_prob_dis1 = t_probs.detach().numpy()[:generated_num]
        for i in nll_prob_dis1:
            if i[0] == 0:
                continue
            nll_prob_dis.append(i[0])
        nll_prob_dis2 = t_probs.detach().numpy()[-generated_num:]
        for i in nll_prob_dis2:
            if i[1] == 0:
                continue
            nll_prob_dis.append(i[1])
        dis_loss = np.average(np.log(nll_prob_dis)) * -1

        nll_prob_gen1 = t_probs.detach().numpy()[-generated_num:]
        nll_prob_gen = []
        for i in nll_prob_gen1:
            if i[0] == 0:
                continue
            nll_prob_gen.append(i[0])

        gen_loss = np.average(np.log(nll_prob_gen)) * -1
        reward = np.average(nll_prob_gen)
        print('nll_prob_gen', nll_prob_gen)
        print('pre loss reward: ', reward)
        # # t_probs = t_probs[:, 1]
        # log_probs = torch.log(t_probs)
        # nll = nn.NLLLoss()
        y_true = torch.tensor(y_true, dtype=torch.long)
        # nll_prob_gen = log_probs[-100:]
        # dis_loss = nll(log_probs, y_true)
        # gen_loss = np.average(nll_prob_gen[:, 0].detach().numpy()) * -1
        t_probs = t_probs.detach().numpy()
        t_auc = metrics.roc_auc_score(y_true, t_probs[:, 1])
        # print(t_probs)
    return t_auc, dis_loss, gen_loss, reward


def main(model_name, dga, exp_num, use, cuda, dis_num):
    random.seed(SEED)
    np.random.seed(SEED)

    # Pretrain Discriminator
    discriminator = exp1.classifier(model_name, exp_num, dga, cuda, dis_num)
    # Define Networks
    generator = Generator(VOCAB_SIZE, g_emb_dim, g_hidden_dim, opt.cuda)
    auc = []
    gen_loss = []
    dis_loss = []
    reward_all = []
    # Generate toy data using target lstm
    print('Generating data ...')
    real_samples()
    # Load data from file
    gen_data_iter = GenDataIter(POSITIVE_FILE, BATCH_SIZE)
    # Pretrain Generator using MLE
    gen_criterion = nn.NLLLoss(reduction='sum')
    gen_optimizer = optim.Adam(generator.parameters())
    if opt.cuda:
        discriminator = discriminator.cuda()
        gen_criterion = gen_criterion.cuda()
        generator = generator.cuda()
    print('Pretrain with MLE ...')
    for epoch in range(PRE_EPOCH_NUM):
        loss = train_epoch(generator, gen_data_iter, gen_criterion, gen_optimizer)
        print('Epoch [%d] Model Loss: %f' % (epoch, loss))
        t_auc, dis_loss_i, gen_loss_i, reward_i = AUC(generator, discriminator, model_name, 4096)
        print('Epoch', epoch, t_auc, dis_loss_i, gen_loss_i, reward_i)
        print(auc, dis_loss, gen_loss)
        auc.append(t_auc)
        dis_loss.append(dis_loss_i)
        gen_loss.append(gen_loss_i)
        reward_all.append(reward_i)
        # generate_samples(generator, BATCH_SIZE, GENERATED_NUM, EVAL_FILE)
    # Adversarial Training
    rollout = Rollout(generator, 0.7)
    print('#####################################################')
    print('Start Adeversatial Training...\n')
    gen_gan_loss = GANLoss()
    gen_gan_optm = optim.Adam(generator.parameters())
    if opt.cuda:
        gen_gan_loss = gen_gan_loss.cuda()
    for total_batch in range(TOTAL_BATCH):
        # Train the generator for one step
        samples = generator.sample(BATCH_SIZE, g_sequence_len)
        # construct the input to the genrator, add zeros before samples and delete the last column
        zeros = torch.zeros((BATCH_SIZE, 1)).type(torch.LongTensor)
        if samples.is_cuda:
            zeros = zeros.cuda()
        inputs = Variable(torch.cat([zeros, samples.data], dim=1)[:, :-1].contiguous())
        targets = Variable(samples.data).contiguous().view((-1,))
        # calculate the reward
        rewards = rollout.get_reward(samples, reward_num, discriminator, alphabet, model_name, use, cuda)
        # rewards = rollout.get_reward(samples, 2, discriminator, alphabet, model_name)
        rewards = Variable(torch.Tensor(rewards))
        if opt.cuda:
            rewards = torch.exp(rewards.cuda()).contiguous().view((-1,))
        prob = generator.forward(inputs)
        loss = gen_gan_loss(prob, targets, rewards)
        gen_gan_optm.zero_grad()
        loss.backward()
        gen_gan_optm.step()
        t_auc, dis_loss_i, gen_loss_i, reward_i = AUC(generator, discriminator, model_name, 4096)
        print('Epoch[%d]: auc is %f gen_loss is %f dis_loss is %f' % (total_batch, t_auc, gen_loss_i, dis_loss_i))
        print(auc, dis_loss, gen_loss)
        auc.append(t_auc)
        dis_loss.append(dis_loss_i)
        gen_loss.append(gen_loss_i)
        reward_all.append(reward_i)
        # if auc < 0.5:
        #     break

        rollout.update_params()

        # samples = []
        # for _ in range(int(4096 / BATCH_SIZE)):
        #     sample = generator.sample(BATCH_SIZE, g_sequence_len).cpu().data.numpy().tolist()
        #     samples.extend(sample)
        # new_str = []
        # for s in samples:
        #     new_int = []
        #     for b in s:
        #         if (int_to_char[b] == ' '):
        #             continue
        #         new_int.append(int_to_char[b])
        #     str1 = ''.join(new_int)
        #     new_str.append(str1)
        # benign = []
        # benign_b = file_check.get_no_dot(benign_txt)
        # resultlist = random.sample(range(0, len(benign_b)), 4096)
        # for i in resultlist:
        #     benign.append(benign_b[i])
        #
        # y_true = [0] * len(benign) + [1] * len(new_str)
        # data = benign + new_str
        # # discriminator = cnn.retrain(discriminator, data, y_true, False)
        # feature, dictionary = wordgraph.create_graph(data)
        # train_x, train_y = wordgraph.get_data('C:/Users/sxy/Desktop/experiment/graph/graph.txt')
        # # 实例化模型，划分规则用的是熵
        # clf = tree.DecisionTreeClassifier(criterion="entropy")
        # # 拟合数据
        # clf = clf.fit(train_x, train_y)
        # prob = clf.predict(feature)
        # discriminator = wordgraph.Wordgraph(dictionary, prob)

        # t_auc, dis_loss_i, gen_loss_i = AUC(generator, discriminator, model_name, 4096)
        # print('Retrain: Epoch[%d]: auc is %f gen_loss is %f dis_loss is %f' % (total_batch, t_auc, gen_loss_i, dis_loss_i))
        # print(auc, dis_loss, gen_loss)
        # auc.append(t_auc)
        # dis_loss.append(dis_loss_i)
        # gen_loss.append(gen_loss_i)
    # save model
    # if exp_num is 1:
    #     if use is 'collision':
    #         torch.save(generator.state_dict(), './model/exp1/' + str(model_name) + '-' + str(use) + '.trc')
    #     else:
    #         torch.save(generator.state_dict(), './model/exp1/' + str(model_name) + '.trc')
    # else:
    #     torch.save(generator.state_dict(), './model/exp2/' + str(model_name) + '-' + str(dga) + '.trc')
    print(auc)
    print(gen_loss)
    print(dis_loss)
    f = open("./" + str(model_name) + "loss.txt", 'a')
    for i in auc:
        f.write(str(i) + ' ')
    f.write('\n')
    for i in gen_loss:
        f.write(str(i) + ' ')
    f.write('\n')
    for i in dis_loss:
        f.write(str(i) + ' ')
    f.write('\n')
    for i in reward_all:
        f.write(str(i) + ' ')
    f.write('\n')
    f.close()


if __name__ == '__main__':
    model_name = ['LSTM']
    for i in range(10, 32, 10):
        for model in model_name:
            # experiment 1
            main(model, 'no', 1, 'mmm', opt.cuda, i)
            # main(model, 'no', 1, 'collision', opt.cuda)
            # experiment 2
            # DGA_train = ['matsnu', 'gozi', 'suppobox', 'pykspa', 'khaos', 'charbot']
            # for dga in DGA_train:
            #     main(model, dga, 2)
