# -*- coding:utf-8 -*-

import os
import random
import math
import copy

import tqdm

import numpy as np

import torch
from tensorflow.keras.preprocessing import sequence
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
alphabet = [' ', 'g', 'o', 'l', 'e', 'y', 'u', 't', 'b', 'm', 'a', 'i', 'd', 'q', 's', 'h', 'f', 'c', 'k', '3', '6',
            '0', 'j', 'z', 'n', 'w', 'p', 'r', 'x', 'v', '1', '8', '7', '2', '9', '-', '5', '4', '_']
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
class Rollout(object):
    """Roll-out policy"""
    def __init__(self, model, update_rate):
        self.ori_model = model
        self.own_model = copy.deepcopy(model)
        self.update_rate = update_rate

    def get_back(self, samples):
        new_str = []
        for s in samples.cpu().detach().numpy():
            new_int = []
            for b in s:
                if (int_to_char[b] == ' '):
                    continue
                new_int.append(int_to_char[b])
            str1 = ''.join(new_int)
            new_str.append(str1)
        X = [[char_to_int[y] for y in x] for x in new_str]
        X = sequence.pad_sequences(X, 100)
        X = torch.tensor(X, dtype=torch.long).cuda()
        return X

    def get_reward(self, x, num, discriminator):
        """
        Args:
            x : (batch_size, seq_len) input data
            num : roll-out number
            discriminator : discrimanator model
        """
        rewards = []
        batch_size = x.size(0)
        seq_len = x.size(1)
        for i in range(num):
            for l in range(1, seq_len):
                data = x[:, 0:l]
                samples = self.own_model.sample(batch_size, seq_len, data)
                samples = self.get_back(samples)
                pred = discriminator(samples)
                # print(pred)
                pred = pred.cpu().data[:, 0].numpy()
                if i == 0:
                    rewards.append(pred)
                else:
                    rewards[l - 1] += pred

            # for the last token
            x = self.get_back(x)
            pred = discriminator(x)
            pred = pred.cpu().data[:, 0].numpy()
            if i == 0:
                rewards.append(pred)
            else:
                rewards[seq_len - 1] += pred
        rewards = np.transpose(np.array(rewards)) / (1.0 * num)  # batch_size * seq_len
        return rewards

    def update_params(self):
        dic = {}
        for name, param in self.ori_model.named_parameters():
            dic[name] = param.data
        for name, param in self.own_model.named_parameters():
            if name.startswith('emb'):
                param.data = dic[name]
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]
