import torch
import numpy as np
from torch.utils.data import Dataset
import torch.utils.data as data_utils

n_gram = dict()
Sample_Num = 10000


def get_domain():
    data = []
    lines = []
    with open("benign.txt", "r") as f:  # 打开文件
        data = f.read().splitlines()
        # line = f.readline()  # 读取文件
        # while line:
        #     line_split = line.split('.')
        #     data.append(line_split[0])
        #     line = f.readline()
        #     lines.append(''.join(line))
    return data[:Sample_Num]


def get_lines():
    lines = []
    with open("data_10000.txt", "r") as f:  # 打开文件
        line = f.readline()  # 读取文件
        while line:
            line_split = line.split('.')
            lines.append(','.join(line_split[0]))
            line = f.readline()
    return lines


def n_gram_generator(name):
    for i in range(4):
        current = 0
        while current+i+1 < len(name)+1:
            gram = name[current:current+i+1]
            current += 1
            if gram in n_gram:
                n_gram[gram] += 1
            else:
                n_gram[gram] = 1


def generate_dictory():
    data = get_domain()
    for i in data:
        n_gram_generator(i)
    result = sorted(n_gram.items(), key=lambda x: x[1], reverse=True)  # x[0]是字典的键，x[1]是字典的值
    # 打印输出排序后的结果
    my_dictory = []
    my_dictory.append(' ')
    for i in result:
        my_dictory.append(i[0])
    return my_dictory[:5000]


def token_domain_forward(name, m_dictory):
    current = 0
    token = []
    while current < len(name):
        if (len(name)-current) > 4:
            num = 4
        else:
            num = len(name)-current
        for i in range(num, -1, -1):
            if name[current:current+i+1] in m_dictory:
                token.append(name[current:current+i+1])
                current = current+i+1
                break
    return token


def token_domain_backward(name, m_dictory):
    current = len(name)
    token = []
    while current > 0:
        if current > 3:
            num = 4
        else:
            num = current
        for i in range(num, -1, -1):
            if name[current-i:current] in m_dictory:
                token.append(name[current-i:current])
                current = current-i
                break
    return token


def individual_num(name):
    num = 0
    for i in name:
        if len(i) == 1:
           num += 1
    return num


def Bi_direction_Maximum_Matching(forward, backward):
    if len(forward) > len(backward):
        return backward
    elif len(forward) < len(backward):
        return forward
    else:
        if forward == backward:
            return forward
        else:
            if individual_num(forward) > individual_num(backward):
                return backward
            elif individual_num(forward) < individual_num(backward):
                return forward
            else:
                return forward


def Ont_hot():
    domain = get_domain()
    exp_dictory = generate_dictory()
    one_hot = []
    data1 = []
    max_length = 0
    for i in domain:
        forward = token_domain_forward(i, exp_dictory)
        backward = token_domain_backward(i, exp_dictory)
        data = Bi_direction_Maximum_Matching(forward, backward)
        if len(data) > max_length:
            max_length = len(data)
        # print(data)
        data1.append(data)
        char_to_int = dict((c, i)for i, c in enumerate(exp_dictory))
        integer_encoded = [char_to_int[char] for char in data]
        i = len(integer_encoded)
        while i < 10:
            integer_encoded.append(0)
            i += 1
        s = (10, 5000)
        # label = torch.LongTensor(integer_encoded)
        one_hot1 = np.zeros(s, dtype=torch.LongTensor)
        for (j, k) in zip(one_hot1, integer_encoded):
            j[k] = 1
        # one_hot1 = one_hot1.T
        one_hot.append(one_hot1)
    return one_hot, max_length, exp_dictory
    # one_hot = torch.Tensor(one_hot)


def int_to_char(samples):
    exp_dictory = generate_dictory()
    int_to_char = dict((i, c) for i, c in enumerate(exp_dictory))
    new_strs = []
    for sample in samples:
        str = []
        for i in sample:
            if int_to_char[i] == ' ':
                continue
            str.append(int_to_char[i])
        str1 = ''.join(str)
        new_strs.append(str1)
    print("new str")
    print(new_strs[0])
    result = np.array(new_strs)
    np.savetxt('khaos_original_110000.txt', result, fmt='%s')
    return new_strs


# 继承 Dataset类 ，将自己的数据构造成dataloader可以处理处理的形式
class DealDataset(Dataset):

    def __init__(self):
        train_data = Ont_hot()
        self.x_data = train_data
        # self.y_data = train_y
        self.len = Sample_Num

    def __getitem__(self, index):
        return self.x_data[index]  # , self.y_data[index]

    def __len__(self):
        return self.len


def get_training_loader():
    train_set = DealDataset()
    assert train_set
    train_loader = data_utils.dataloader(train_set, batch_size=64, shuffle=True)
    return train_loader
# train_set：上面自己定义的数据类
# batch_size=32：实现批量读取数据，比如一次取32个数据
# shuffle=True：将顺序打乱
# collate_fn：表示的是如何读取样本，可以自己定义函数来准确的说明想要实现的功能。


if __name__ == '__main__':
    a, b = Ont_hot()
    # one_hot = Ont_hot()
    # onehot = torch.Tensor(one_hot)
    # a = np.array(one_hot)
    # onehot = torch.from_numpy(a)






