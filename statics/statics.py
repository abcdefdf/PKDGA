import torch.nn.functional as F
from tqdm import tqdm
import sklearn.metrics as metrics
import random
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import KFold
import math
import torch
import pandas as pd

# 0 is benign; 1 is malicious
benign_txt = "C:/Users/sxy/Desktop/experiment/dataset/train/benign.txt"
malicious_txt = 'C:/Users/sxy/Desktop/experiment/dataset/train/malicious.txt'


def bigram_set(domain):
    k = 0
    bigram = []
    while k < len(domain) - 1:
        bigram.append(domain[k:k + 2])
        k += 1
    return list(set(bigram))


def JI(d1, d2):
    # bigrams
    c1 = bigram_set(d1)
    c2 = bigram_set(d2)
    inter = list(set(c1).intersection(set(c2)))
    un = list(set(c1).union(set(c2)))
    if len(un) is 0:
        return 0
    return 1 - len(inter) / len(un)


def get_one_un_bigram(name, dict_unigram, dict_bigram):
    for i in name.lower():
        dict_unigram[i] += 1
    k = 0
    while k < len(name)-1:
        bigram_temp = name[k:k + 2]
        dict_bigram[bigram_temp] = dict_bigram.get(bigram_temp, 0) + 1
        k += 1
    return dict_unigram, dict_bigram


def unigram_bigram(name, type):
    alpha = 'abcdefghijklmnopqrstuvwxyz1234567890_-'
    dict_unigram = {}
    dict_bigram = {}
    for i in alpha:
        dict_unigram[i] = 0
    if type is 'file':
        f = open(name, 'r')
        data = f.read().splitlines()
        f.close()
        for i in data:
            dict_unigram, dict_bigram = get_one_un_bigram(i, dict_unigram, dict_bigram)
    else:
        dict_unigram, dict_bigram = get_one_un_bigram(name, dict_unigram, dict_bigram)

    count_unigram = sum(dict_unigram.values())
    unigram_distribution = [j/count_unigram for i, j in dict_unigram.items()]
    count_bigram = sum(dict_bigram.values())
    bigram_distribution = [j/count_bigram for i, j in dict_bigram.items()]
    return unigram_distribution, bigram_distribution


def KL_SYM(domain1, domain2):
    domain1_unigram, _ = unigram_bigram(domain1, 'domain')
    domain2_unigram, _ = unigram_bigram(domain2, 'domain')

    domain1_unigram = F.softmax(torch.tensor(domain1_unigram))
    domain2_unigram = F.softmax(torch.tensor(domain2_unigram))
    kl1 = 0.0
    kl2 = 0.0
    for i in range(0, len(domain1_unigram)):
        if domain1_unigram[i] != 0.0 and domain2_unigram[i] != 0.0:
            kl1 += domain1_unigram[i] * math.log(domain1_unigram[i]/domain2_unigram[i])
            kl2 += domain2_unigram[i] * math.log(domain2_unigram[i]/domain1_unigram[i])
    return 0.5 * (kl1 + kl2)


def ED(str1, str2):
    """
    计算字符串 str1 和 str2 的编辑距离
    :param str1
    :param str2
    :return:
    """
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]

    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)
    return matrix[len(str1)][len(str2)]


class statics_classifier:
    def __init__(self, benign, malicious):
        self.benign = benign
        self.malicious = malicious

    def predict_proba(self, domain, train_num=9999):
        prob = []
        proba = []
        for domain_i in tqdm(domain):
            ji_benign = 0.0
            ji_malicious = 0.0
            p = []
            result_benign = random.sample(range(0, len(self.benign)), train_num)
            for i in result_benign:
                ji = JI(domain_i, self.benign[i])
                # ji = ED(domain_i, self.benign[i])
                # ji = KL_SYM(domain_i, self.benign[i])
                ji_benign += ji
            result_malicious = random.sample(range(0, len(self.malicious)), train_num)
            for i in result_malicious:
                ji = JI(domain_i, self.malicious[i])
                # ji = ED(domain_i, self.malicious[i])
                # ji = KL_SYM(domain_i, self.malicious[i])
                ji_malicious += ji
            p.append(ji_benign/train_num)
            p.append(ji_malicious/train_num)
            # print(ji_benign/len(self.benign), ji_malicious/len(self.malicious))
            if ji_benign / train_num < ji_malicious / train_num:
                prob.append(0)
            else:
                prob.append(1)
            proba.append(ji_benign/train_num / ji_malicious/train_num)
        return prob, proba

    def KL_Our(self, domain, train_num=10000):
        prob = []
        proba = []
        for domain_i in tqdm(domain):
            kl_benign_min = 0.0
            kl_malicious_min = 0.0
            kl_benign = 0.0
            kl_malicious = 0.0
            result_benign = random.sample(range(0, len(self.benign)), train_num)
            for i in result_benign:
                kl = KL_SYM(domain_i, self.benign[i])
                kl_benign += kl
                if kl < kl_benign_min:
                    kl_benign_min = kl
            result_malicious = random.sample(range(0, len(self.malicious)), train_num)
            for i in result_malicious:
                kl = KL_SYM(domain_i, self.malicious[i])
                kl_malicious += kl
                if kl < kl_malicious_min:
                    kl_malicious_min = kl

            if kl_benign / train_num < kl_malicious / train_num:
                prob.append(0)
            else:
                prob.append(1)
        return prob

    def ED_our(self, domain, train_num=10000):
        prob = []
        for domain_i in tqdm(domain):
            ed_benign = 0.0
            ed_malicious = 0.0
            result_benign = random.sample(range(0, len(self.benign)), train_num)
            for i in result_benign:
                ed = ED(domain_i, self.benign[i])
                ed_benign += ed
            result_malicious = random.sample(range(0, len(self.malicious)), train_num)
            for i in result_malicious:
                ed = ED(domain_i, self.malicious[i])
                ed_malicious += ed

            if ed_benign / train_num < ed_malicious / train_num:
                prob.append(0)
            else:
                prob.append(1)
        return prob


def exp1():
    f = open(benign_txt, 'r')
    benign_original = f.read().splitlines()
    f.close()
    f = open(malicious_txt, 'r')
    malicious_original = f.read().splitlines()
    f.close()
    index = [i for i in range(len(benign_original))]
    random.shuffle(index)
    benign = []
    malicious = []
    for i in index:
        benign.append(benign_original[i])
        malicious.append(malicious_original[i])
    # test_benign = benign[-1000:]
    # test_malicious = malicious[-1000:]
    f_new = open('C:/Users/sxy/Desktop/experiment/Result/new/exp1-statics.txt', 'w')
    dis = statics_classifier(benign, malicious)
    dga = ['maskDGA']
    # dga = ['our', 'khaos_original', 'kraken', 'gozi', 'suppobox']
    for i in dga:
        if i == 'our':
            f = open('C:/Users/sxy/Desktop/experiment/dataset/our/exp1-CNN-test.txt')
        else:
            f = open('C:/Users/sxy/Desktop/experiment/dataset/test/' + str(i) + '-test.txt')
        malicious_test = f.read().splitlines()
        f.close()
        f = open('C:/Users/sxy/Desktop/experiment/dataset/test/benign-test.txt', 'r')
        benign_test = f.read().splitlines()
        f.close()
        f_new.write(str(i) + '\n')
        index = [i for i in range(1000)]
        random.shuffle(index)
        benign_test1 = []
        malicious_test1 = []
        for k in index:
            benign_test1.append(benign_test[k])
            malicious_test1.append(malicious_test[k])
        data = benign_test1 + malicious_test1
        y_true = [0] * len(benign_test1) + [1] * len(malicious_test1)
        prob, proba = dis.predict_proba(data)
        t_auc = metrics.roc_auc_score(y_true, proba)
        f_new.write(str(i) + ': ' + str(t_auc) + '\n')
        for i in proba:
            f_new.write(str(i) + ' ')
        f_new.write('\n')
        for i in y_true:
            f_new.write(str(i)+ ' ')
        # f_new.write('auc: ' + str(t_auc) + '\n')
        # fpr, tpr, threshold = metrics.roc_curve(y_true, proba)
        # test = pd.DataFrame({'1': y_true, '2': proba})
        # test.to_csv('C:/Users/sxy/Desktop/experiment/Result/new/exp1_statics.csv')
        # f1 = open('C:/Users/sxy/Desktop/experiment/Result/new/exp1_statics.csv', 'a', newline='')
        # # plt.title('%s auc = %f' % (i, t_auc))
        # # if t_auc < 0.5:
        # #     prob = []
        # #     for i in proba:
        # #         prob.append(1/i)
        # #     proba = prob
        # #     t_auc = metrics.roc_auc_score(y_true, proba)
        # writer = csv.writer(f1)
        # writer.writerow(y_true)
        # writer.writerow(proba)
        # print('%s: auc :%f' % (i, t_auc))
        # writer.writerow(y_true)
        #
        # writer.writerow(fpr)
        # writer.writerow(tpr)
        # plt.plot(fpr, tpr, 'b', label='Val AUC = %0.3f' % t_auc)
        # plt.show()


def exp2():
    dga_train = ['maskDGA']
    # dga_train = ['khaos_original', 'kraken', 'gozi', 'suppobox', 'maskDGA', 'our']
    for dag_malicious in dga_train:
        if dag_malicious == 'our':
            f = open('C:/Users/sxy/Desktop/experiment/dataset/our/1/exp1-LSTM-test.txt')
        else:
            f = open('C:/Users/sxy/Desktop/experiment/dataset/statics/' + str(dag_malicious) + '-train.txt')
        malicious_original = f.read().splitlines()
        f.close()
        f = open('C:/Users/sxy/Desktop/experiment/dataset/train/exp2/benign-train.txt', 'r')
        benign_original = f.read().splitlines()
        f.close()
        index = [i for i in range(10000)]
        random.shuffle(index)
        benign = []
        malicious = []
        for i in index:
            benign.append(benign_original[i])
            malicious.append(malicious_original[i])
        # test_benign = benign[-1000:]
        # test_malicious = malicious[-1000:]
        f_new = open('C:/Users/sxy/Desktop/experiment/Result/new/exp2-statics.txt', 'a')
        dis = statics_classifier(benign, malicious)
        dga = ['khaos_original', 'kraken', 'gozi', 'suppobox', 'maskDGA', 'our']
        # dga = ['khaos_lstm', 'khaos_original', 'kraken', 'charbot', 'gozi', 'suppobox', 'pykspa', 'our']
        for i in dga:
            if i == 'our':
                f = open('C:/Users/sxy/Desktop/experiment/dataset/test/1000/exp1-LSTM-test.txt')
            elif i == 'maskDGA' or i == 'khaos_original':
                f = open('C:/Users/sxy/Desktop/experiment/dataset/test/' + str(i) + '-test.txt')
            else:
                f = open('C:/Users/sxy/Desktop/experiment/dataset/test/1000/' + str(i) + '-test.txt')
            malicious_test = f.read().splitlines()
            f.close()
            f = open('C:/Users/sxy/Desktop/experiment/dataset/test/1000/benign-test.txt')
            benign_test = f.read().splitlines()
            f.close()
            f_new.write(str(dag_malicious) + ' ' + str(i) + '\n')
            index = [i for i in range(1000)]
            random.shuffle(index)
            benign_test1 = []
            malicious_test1 = []
            for k in index:
                benign_test1.append(benign_test[k])
                malicious_test1.append(malicious_test[k])
            data = benign_test1 + malicious_test1
            y_true = [0] * len(benign_test1) + [1] * len(malicious_test1)
            prob, proba = dis.predict_proba(data)
            t_auc = metrics.roc_auc_score(y_true, proba)
            print('train', dag_malicious, 'test', i, t_auc)
            # for j in proba:
            #     f_new.write(str(j) + ' ')
            # f_new.write('\n')
            print('%s: auc :%f' % (i, t_auc))
            f_new.write('auc: ' + str(t_auc) + '\n')
            fpr, tpr, threshold = metrics.roc_curve(y_true, proba)
            f1 = open('saticis_exp2.csv', 'a', newline='')
            writer = csv.writer(f1)
            writer.writerow(fpr)
            writer.writerow(tpr)
            plt.title('%s auc = %f' % (i, t_auc))
            plt.plot(fpr, tpr, 'b', label='Val AUC = %0.3f' % t_auc)
            plt.show()


if __name__ == '__main__':
    exp2()
