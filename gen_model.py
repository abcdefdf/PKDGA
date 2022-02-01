# -*- coding:utf-8 -*-
import lstm.lstm_class as lstm_c
import exp1
import lstm
import data_init
import random
import math
import sklearn.metrics as metrics
from tensorflow.keras.preprocessing import sequence
import argparse
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
# ================== Parameter Definition =================

parser = argparse.ArgumentParser(description='Training Parameter')
parser.add_argument('--cuda', action='store', default=None, type=int)
opt = parser.parse_args()
print(opt)

# Basic Training Paramters
# SEED = 108
BATCH_SIZE = 64
GENERATED_NUM = 9984
POSITIVE_FILE = 'real.data'
NEGATIVE_FILE = 'gene.data'
EVAL_FILE = 'eval.data'
VOCAB_SIZE = 39
# TOTAL_BATCH = 20
TOTAL_BATCH = 0
# PRE_EPOCH_NUM = 100
PRE_EPOCH_NUM = 1
reward_num = 20
# reward_num = 2
if opt.cuda is not None and opt.cuda >= 0:
    torch.cuda.set_device(opt.cuda)
    opt.cuda = True

# Genrator Parameters
g_emb_dim = 32
g_hidden_dim = 64
# g_sequence_len = 100
g_sequence_len = 65

benign_txt = 'C:/Users/sxy/Desktop/experiment/dataset/train/benign.txt'
malicious_txt = 'C:/Users/sxy/Desktop/experiment/dataset/train/malicious.txt'

# alphabet = [' ', 'g', 'o', 'l', 'e', 'y', 'u', 't', 'b', 'm', 'a', 'i', 'd', 'q', 's', 'h', 'f', 'c', 'k', '3', '6',
#             '0', 'j', 'z', 'n', 'w', 'p', 'r', 'x', 'v', '1', '8', '7', '2', '9', '-', '5', '4', '_']
#
# int_to_char = dict((i, c) for i, c in enumerate(alphabet))
# char_to_int = dict((c, i) for i, c in enumerate(alphabet))


def generate_samples(model, batch_size, generated_num, output_file):
    samples = []
    for _ in range(int(generated_num / batch_size)):
        sample = model.sample(batch_size, g_sequence_len).cpu().data.numpy().tolist()
        samples.extend(sample)
    with open(output_file, 'w') as fout:
        for sample in samples:
            string = ' '.join([str(s) for s in sample])
            fout.write('%s\n' % string)
    return samples


def reward_train(dis, gen):
    samples = []
    for _ in range(int(9984 / 64)):
        sample = gen.sample(64, g_sequence_len).cpu().data.numpy().tolist()
        samples.extend(sample)
    samples = torch.LongTensor(np.array(samples))
    rewards = lstm_c.predict_proba(dis, samples)
    rewards = rewards[:, 0].detach().numpy()
    return np.sum(rewards)


def train_epoch(model, data_iter, criterion, optimizer, dis, gen):
    total_loss = 0.
    total_words = 0.
    reward = []
    for (data, target) in data_iter:#tqdm(
        #data_iter, mininterval=2, desc=' - Training', leave=False):
        reward.append(reward_train(dis, gen))
        print(reward)
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
    return math.exp(total_loss / total_words), reward


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


def real_samples(start, end):
    f = open(benign_txt, 'r')
    data = f.read().splitlines()
    f.close()
    data = data[start:end]
    one_hot = []
    for single_data in data:
        # integer encode input data
        integer_encoded = [data_init.char_to_int[char] for char in single_data]
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
            if (data_init.int_to_char[b] == ' '):
                continue
            new_int.append(data_init.int_to_char[b])
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
    elif name is 'Statics':
        t_prob = dis.predict_proba(data, 10000)
        t_auc = metrics.roc_auc_score(y_true, t_prob)
    else:
        X = [[data_init.char_to_int[y] for y in x] for x in data]
        # X = sequence.pad_sequences(X, 100)
        X = sequence.pad_sequences(X, g_sequence_len)
        X = torch.LongTensor(X)
        t_probs = dis.predict_proba(X)
        t_probs = t_probs.detach().numpy()
        t_auc = metrics.roc_auc_score(y_true, t_probs[:, 1])
    return t_auc


def gen_n_samples(gen, file, generated_num=10000, batch_size=128):
    f = open(file, 'a')
    samples = []
    for _ in range(int(generated_num / batch_size)):
        sample = gen.sample(batch_size, g_sequence_len).cpu().data.numpy().tolist()
        samples.extend(sample)
    for s in samples:
        new_int = []
        for b in s:
            if (data_init.int_to_char[b] == ' '):
                continue
            new_int.append(data_init.int_to_char[b])
        str1 = ''.join(new_int)
        f.write(str1+'\n')


def get_reward(x, discriminator):
    """
    Args:
        x : (batch_size, seq_len) input data
        num : roll-out number
        discriminator : discriminator model
    """
    rewards = lstm_c.predict_proba(discriminator, x)
    rewards = rewards[:, 0].detach().numpy()
    return np.sum(rewards)


def main(model_name, dga, exp_num, use, cuda, start, end, dis_num):
    # random.seed(SEED)
    # np.random.seed(SEED)
    # Pretrain Discriminator
    # discriminator = lstm.lstm_class.Discriminator(2, len(data_init.alphabet), 32, 32, False, 1)
    # discriminator.load_state_dict(torch.load('model/lstm.pt'))
    discriminator = exp1.classifier(model_name, exp_num, dga, cuda, dis_num)
    # Define Networks
    generator = Generator(VOCAB_SIZE, g_emb_dim, g_hidden_dim, opt.cuda)

    # Generate toy data using target lstm
    print('Generating data ...')
    real_samples(start, end)
    # Load data from file
    gen_data_iter = GenDataIter(POSITIVE_FILE, BATCH_SIZE)
    # Pretrain Generator using MLE
    gen_criterion = nn.NLLLoss(reduction='sum')
    gen_optimizer = optim.Adam(generator.parameters())
    if opt.cuda:
        discriminator = discriminator.cuda()
        gen_criterion = gen_criterion.cuda()
        generator = generator.cuda()

    # f = open('reward_pic/mle.txt', 'a')
    reward_all = []
    print('Pretrain with MLE ...')
    for epoch in range(PRE_EPOCH_NUM):
        loss, reward = train_epoch(generator, gen_data_iter, gen_criterion, gen_optimizer, discriminator, generator)
        f_mle = open('reward_pic/mle.txt', 'a')
        f_mle.write('mle ' + str(start) + ' ' + str(end) + '\n')
        f_mle.write(str(reward))
        f_mle.close()
        print('reward', reward)
        print('Epoch [%d] Model Loss: %f' % (epoch, loss))
        samples = generate_samples(generator, BATCH_SIZE, GENERATED_NUM, EVAL_FILE)
        # print(samples)
        # samples = np.array([np.array(i) for i in samples])
        # print(samples.size())
        # print(samples.size(0))
        rewards = get_reward(torch.LongTensor(np.array(samples)), discriminator)
        reward_all.append(rewards)
        # f.write(str(rewards) + '\n')
        print(reward_all)

    auc = AUC(generator, discriminator, model_name, 4096)
    # gen_n_samples(generator, 'adverse_samples/mle.txt')
    print('Epoch MLE: auc is %f' % auc)

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
        rewards = rollout.get_reward(samples, reward_num, discriminator, data_init.alphabet, model_name, use, cuda)
        rewards_ave = np.sum(rewards.copy())
        # f.write(str(rewards_ave) + '\n')
        reward_all.append(rewards_ave)
        print(reward_all)
        # rewards = rollout.get_reward(samples, 2, discriminator, alphabet, model_name)
        rewards = Variable(torch.Tensor(rewards))
        if opt.cuda:
            rewards = torch.exp(rewards.cuda()).contiguous().view((-1,))
        prob = generator.forward(inputs)
        loss = gen_gan_loss(prob, targets, rewards)
        gen_gan_optm.zero_grad()
        loss.backward()
        gen_gan_optm.step()
        auc = AUC(generator, discriminator, model_name, 4096)
        # gen_n_samples(generator, 'adverse_samples/12_20_no_mle/' + str(total_batch) + '.txt')
        print('Epoch[%d]: auc is %f' % (total_batch, auc))
        # if auc < 0.5:
        #     break

        rollout.update_params()
    # f.close()
    # save model
    # if exp_num is 1:
    #     if use is 'collision':
    #         torch.save(generator.state_dict(), './model/exp1/' + str(model_name) + '-' + str(use) + '.trc')
    #     else:
    #         torch.save(generator.state_dict(), './model/exp1/' + str(model_name) + '_30.trc')
    # else:
    #     torch.save(generator.state_dict(), './model/exp2/' + str(model_name) + '-' + str(dga) + '.trc')


if __name__ == '__main__':
    model_name = ['LSTM']
    s = [30000, 40000, 50000, 60000, 70000, 80000]
    dis_num = [30, 40, 50, 60, 70, 80]
    # for start in [20000, 30000, 40000, 50000, 60000, 70000, 80000]:
    for start in range(6):
        for model in model_name:
            # experiment 1
            # main(model, 'no', 1, 'mmm', opt.cuda)
            main(model, 'no', 1, 'no', opt.cuda, s[start], s[start]+9984, dis_num[start])
            # experiment 2
            # DGA_train = ['matsnu', 'gozi', 'suppobox', 'pykspa', 'khaos', 'charbot']
            # for dga in DGA_train:
            #     main(model, dga, 2)
