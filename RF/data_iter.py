# -*- coding:utf-8 -*-

import os
import random
import math

import torch.nn as nn
from tensorflow.keras.preprocessing import sequence
import numpy as np
import torch


class GenDataIter(object):
    """ Toy data iter to load digits"""
    def __init__(self, data_file, batch_size):
        super(GenDataIter, self).__init__()
        self.batch_size = batch_size
        self.data_lis = self.read_file(data_file)
        self.data_num = len(self.data_lis)
        self.indices = range(self.data_num)
        self.num_batches = int(math.ceil(float(self.data_num)/self.batch_size))
        self.idx = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()
    
    def reset(self):
        self.idx = 0
        random.shuffle(self.data_lis)

    def next(self):
        if self.idx >= self.data_num:
            raise StopIteration
        index = self.indices[self.idx:self.idx+self.batch_size]
        d = [self.data_lis[i] for i in index]
        d = torch.LongTensor(np.asarray(d, dtype='int64'))
        data = torch.cat([torch.zeros(self.batch_size, 1).long(), d], dim=1)
        target = torch.cat([d, torch.zeros(self.batch_size, 1).long()], dim=1)
        self.idx += self.batch_size
        return data, target

    def read_file(self, data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
        lis = []
        for line in lines:
            l = line.strip().split(' ')
            l = [int(s) for s in l]
            lis.append(l)
        # lis = nn.utils.rnn.pad_sequence([torch.from_numpy(np.array(x)) for x in lis + lis], batch_first=True,
        #                                 padding_value=0)
        # lis = sequence.pad_sequences(lis, 100)
        return lis


class DisDataIter(object):
    """ Toy data iter to load digits"""
    def __init__(self, real_data_file, fake_data_file, batch_size):
        super(DisDataIter, self).__init__()
        self.batch_size = batch_size
        real_data_lis = self.read_file_len(real_data_file)
        fake_data_lis = self.read_file_len(fake_data_file)
        # self.data, self.data_len = self.read_file(real_data_file, fake_data_file)
        self.data = self.read_file(real_data_file, fake_data_file)
        self.labels = [0 for _ in range(real_data_lis)] +\
                      [1 for _ in range(fake_data_lis)]
        # self.pairs = list(zip(self.data, self.labels, self.data_len))
        self.pairs = list(zip(self.data, self.labels))
        self.data_num = len(self.pairs)
        self.indices = range(self.data_num)
        self.num_batches = int(math.ceil(float(self.data_num)/self.batch_size))
        self.idx = 0
        self.reset()

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()
    
    def reset(self):
        self.idx = 0
        random.shuffle(self.pairs)

    def next(self):
        if self.idx >= self.data_num:
            raise StopIteration
        index = self.indices[self.idx:self.idx+self.batch_size]
        pairs = [self.pairs[i] for i in index]
        data = [p[0] for p in pairs]
        label = [p[1] for p in pairs]
        # data_len = [p[2] for p in pairs]
        # data = torch.LongTensor([item.numpy() for item in data]).cuda()
        # label = torch.LongTensor([item for item in label]).cuda()
        # data_len = torch.LongTensor([item for item in data_len]).cuda()
        data = torch.LongTensor(np.asarray(data, dtype='int64'))
        label = torch.LongTensor(np.asarray(label, dtype='int64'))
        # data_len = torch.LongTensor(np.asarray(data_len, dtype='int64'))
        self.idx += self.batch_size
        # return data, label, data_len
        return data, label

    def read_file_len(self, data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
        return len(lines)

    def read_file(self, real_file, fake_file):
        with open(real_file, 'r') as f:
            lines = f.readlines()
        lis = []
        # lis_len = []
        for line in lines:
            l = line.strip().split(' ')
            l = [int(s) for s in l]
            # lis_len.append(len(l))
            lis.append(l)

        with open(fake_file, 'r') as f:
            lines_f = f.readlines()
        lis_f = []
        for line in lines_f:
            l_f = line.strip().split(' ')
            l_f = [int(s) for s in l_f]
            lis_f.append(l_f)
            # lis_len.append(len(l_f))
        # lis = nn.utils.rnn.pad_sequence([torch.from_numpy(np.array(x)) for x in lis+lis_f], batch_first=True,
        #                                     padding_value=0)
        # lis = nn.utils.rnn.pack_padded_sequence(torch.tensor(lis+lis_f), lis_len, batch_first=True)
        lis = sequence.pad_sequences(lis+lis_f, 100)
        # return lis, lis_len
        return lis


