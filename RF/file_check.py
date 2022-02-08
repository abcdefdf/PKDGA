from urllib import request
from tqdm import *
import matplotlib.pyplot as plt
import glob
import pandas as pd
import random
import os
import random

request_failure = []


def writeToText(list, fn):
    file = open(fn, 'a')
    for a in list:
        file.write(a + '\n')
    file.close()


def CheckDomainStatus(i, domain, tlds):
    API = "http://panda.www.net.cn/cgi-bin/check.cgi?area_domain="
    not_resign_domain = []
    domain_count = []
    dict = {}
    print('num: ', (i / 100), end="")
    with tqdm(total=len(domain)) as pbar:
        for i in range(len(domain)):
            name = domain[i]
            cout = 0
            for tld in tlds:
                domain_name = name + "." + tld
                url = API + domain_name
                try:
                    xhtml = request.urlopen(url, timeout=5).read().decode()
                except Exception as e:
                    print('Exception', e)
                    continue
                r = xhtml.find(r'<original>210')  # 210 is not resigned, 211 is resigned
                if r != -1:
                    cout += 1
                    # print(domain + 'not resign')
                    not_resign_domain.append(domain_name)
            dict[name] = cout
            pbar.update(1)
    return not_resign_domain, domain_count, dict


def Back_SeqGAN_txt(exp, dga, model):
    global txt
    if exp == 1:
        txt = "C:/Users/sxy/Desktop/4.29/Dataset/exp1/exp1-data-" + str(model) + '.txt'
    elif exp == 2:
        txt = 'C:/Users/sxy/Desktop/4.29/Dataset/exp2/exp2-' + str(model) + '-' + str(dga) + '.txt'
    else:
        print("error")
    return txt


# def Back_txt(name):
#     # if name is "SeqGAN":
#     #     return "C:/Users/sxy/Desktop/Dataset/exp2/test/SeqGAN.txt"
#     if name is "Charbot":
#         return "C:/Users/sxy/Desktop/4.29/Dataset/test/charbot.txt"
#     elif name is "Khaos":
#         return "C:/Users/sxy/Desktop/4.29/Dataset/test/Khaos.txt"
#     elif name is "DeepDGA":
#         return "C:/Users/sxy/Desktop/My/Predict_dataset/DeepDGA_1000.txt"
#     elif name is "Kraken_v1":
#         return "C:/Users/sxy/Desktop/4.29/Dataset/test/kraken_v1.txt"
#     elif name is "Nymaim":
#         return "C:/Users/sxy/Desktop/My/Predict_dataset/Nymaim_1000.txt"
#     elif name is "Pykspa":
#         return "C:/Users/sxy/Desktop/4.29/Dataset/test/pykspa.txt"
#     elif name is "Matsnu":
#         return "C:/Users/sxy/Desktop/4.29/Dataset/test/matsnu.txt"
#     elif name is "Suppobox":
#         return "C:/Users/sxy/Desktop/4.29/Dataset/test/suppobox.txt"
#     elif name is "Gozi":
#         return "C:/Users/sxy/Desktop/4.29/Dataset/test/Gozi.txt"
#     elif name is "Alexa":
#         return "C:/Users/sxy/Desktop/My/Predict_dataset/exp2-36/data_10000.txt"
#     elif name is "all":
#         return "100000_m.txt"
#     else:
#         print("Have no name")

def Back_txt(name):
    # if name is "SeqGAN":
    #     return "C:/Users/sxy/Desktop/Dataset/exp2/train/SeqGAN.txt"
    if name is "Charbot":
        return "C:/Users/sxy/Desktop/My/Predict_dataset/exp2-36/test/charbot.txt"
    elif name is "Khaos":
        return "C:/Users/sxy/Desktop/My/Predict_dataset/exp2-36/test/Khaos.txt"
    elif name is "Kraken_v1":
        return "C:/Users/sxy/Desktop/My/Predict_dataset/exp2-36/test/kraken_v1_1000.txt"
    elif name is "Suppobox":
        return "C:/Users/sxy/Desktop/My/Predict_dataset/exp2-36/test/suppobox_1000.txt"
    elif name is "Gozi":
        return "C:/Users/sxy/Desktop/My/Predict_dataset/exp2-36/test/Gozi_luther_1000.txt"
    else:
        print("Have no name")
# def Back_Train_txt(name):
#     # if name is "SeqGAN":
#     #     return "C:/Users/sxy/Desktop/Dataset/exp2/train/SeqGAN.txt"
#     if name is "Charbot":
#         return "C:/Users/sxy/Desktop/4.29/Dataset/train/charbot-10000.txt"
#     elif name is "Khaos":
#         return "C:/Users/sxy/Desktop/4.29/Dataset/train/Khaos-6914.txt"
#     elif name is "DeepDGA":
#         return "C:/Users/sxy/Desktop/My/Predict_dataset/DeepDGA_1000.txt"
#     elif name is "Kraken_v1":
#         return "C:/Users/sxy/Desktop/4.29/Dataset/train/kraken_v1.txt"
#     elif name is "Nymaim":
#         return "C:/Users/sxy/Desktop/My/Predict_dataset/Nymaim_1000.txt"
#     elif name is "Pykspa":
#         return "C:/Users/sxy/Desktop/4.29/Dataset/train/pykspa.txt"
#     elif name is "Matsnu":
#         return "C:/Users/sxy/Desktop/4.29/Dataset/train/matsnu.txt"
#     elif name is "Suppobox":
#         return "C:/Users/sxy/Desktop/4.29/Dataset/train/suppobox.txt"
#     elif name is "Gozi":
#         return "C:/Users/sxy/Desktop/4.29/Dataset/train/Gozi.txt"
#     elif name is "Alexa":
#         return "C:/Users/sxy/Desktop/My/Predict_dataset/exp2-36/data_10000.txt"
#     elif name is "all":
#         return "100000_m.txt"
#     else:
#         print("Have no name")

def Back_Train_txt(name):
    # if name is "SeqGAN":
    #     return "C:/Users/sxy/Desktop/Dataset/exp2/train/SeqGAN.txt"
    if name is "Charbot":
        return "C:/Users/sxy/Desktop/My/Predict_dataset/exp2-36/train/charbot-10000.txt"
    elif name is "Khaos":
        return "C:/Users/sxy/Desktop/My/Predict_dataset/exp2-36/train/Khaos-6914.txt"
    elif name is "Kraken_v1":
        return "C:/Users/sxy/Desktop/My/Predict_dataset/exp2-36/train/kraken_v1.txt"
    elif name is "Suppobox":
        return "C:/Users/sxy/Desktop/My/Predict_dataset/exp2-36/train/suppobox.txt"
    elif name is "Gozi":
        return "C:/Users/sxy/Desktop/My/Predict_dataset/exp2-36/train/Gozi.txt"
    else:
        print("Have no name")


def get_has_dot(name):
    benign_start = []
    with open(name, "r") as f:
        line = f.readline()
        while line:
            line_split = line.split('.')
            benign_start.append(line_split[0])
            line = f.readline()
    return benign_start


def get_no_dot(name):
    test = []
    with open(name, "r") as f:
        line = f.readline()
        while line:
            test.append(line[:len(line) - 1])
            line = f.readline()
    return test


def drawRoc(roc_auc, fpr, tpr):
    plt.subplots(figsize=(7, 5.5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve(area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


# 域名注册查询
def Check_no_resign(filein):
    word = []
    with open(filein, "r") as f:  # 打开文件
        line = f.readline()  # 读取文件
        while line:
            word.append(line[:len(line) - 1])
            line = f.readline()
    randomresult = random.sample(range(0, len(word)), 100)
    domain = []
    for i in randomresult:
        domain.append(word[i])
    print(domain)
    tlds = ['com', 'cn', 'net', 'org', 'com.cn', 'info', 'cc', 'top', 'wang']
    domain_no_resign, domain_count, dict = CheckDomainStatus(i, domain, tlds)
    print(dict)
    print('no resign len: ', len(domain_no_resign))
    print(domain_no_resign)


# 域名注册奖励
def Check_resign_reward(new_str):
    reward = []
    for name in new_str:
        tlds = ['com', 'cn', 'net', 'org', 'com.cn', 'info', 'cc', 'top', 'wang']
        API = "http://panda.www.net.cn/cgi-bin/check.cgi?area_domain="
        count = 0
        no_result = []
        for tld in tlds:
            domain_name = name + "." + tld
            url = API + domain_name
            try:
                xhtml = request.urlopen(url, timeout=5).read().decode()
            except Exception as e:
                print('Exception', e)
                continue
            r = xhtml.find(r'<original>210')  # 210 is resigned, 211 is no resigned
            if r != -1:
                count += 1
        reward.append(count/len(tlds))
    return reward


# delete len()<3 domain
def Handle_agd(filein, fileout):
    a = get_no_dot(filein)
    with open(fileout, 'w') as f:
        for i in a:
            if len(i) < 3:
                continue
            else:
                f.write(i + '\n')


# Put result(auc/precision/recall) into every-result.txt and generate ROC.png
def to_TXT(name, auc, report, fpr, tpr, txt, lw=2):
    with open('./dataset/every-result.txt', 'a') as f:
        print(name)
        f.write(name + '\n')
        f.write("AUC: " + str(auc) + '\n')
        report = report.splitlines()
        for i in report:
            f.write(i + '\n')
        f.write('\n')
    f.close()

    plt.clf()
    plt.title("%s curve{AUC = %.4f}" % (name, auc))
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.plot(fpr, tpr, color='navy', lw=lw, label='Sat')
    plt.legend(loc="lower right")
    # plt.savefig("./png/png2/" + name + '.png', dpi=75)
    plt.savefig(txt + name + ".png", dpi=75)
    plt.show()


# Distribution of characters
def alphabet_bar(dga=["SeqGAN"]):
    for name in dga:
        x = []
        count = 0
        count_list = []
        interval = []
        file_name = Back_txt(name)
        with open(file_name, 'r') as f:
            fi = f.read()
            fi = fi.lower()
            for i in '-0123456789abcdefghijklmnopqrstuvwxyz':
                interval.append(i)
                c = fi.count(i)
                count += c
                count_list.append(c)
                if c != 0:
                    x.append('{0}:{1}'.format(i, c))
        f.close()
        alphabet_list = [(i * 1.0) / count for i in count_list]

        plt.figure()
        plt.title(name)
        plt.bar(range(1, 38), alphabet_list, bottom=0, width=0.4, label="Number of alphabet", tick_label=interval,
                linewidth=1)
        plt.legend()
        plt.savefig("./png/alphabet/" + name + ".png")
        plt.show()


# Count the number of unregistered per domain name
def statics_no_resign_name(filein):
    result = [0 for i in range(10)]
    with open(filein, 'r') as f:
        line = f.readline()
        while line is not '\n':
            num = line[len(line) - 2]
            print(num)
            result[int(num)] += 1
            line = f.readline()
    f.close()
    print(result)


# Count the number of repeat domain name
def statics_repeat():
    dict = {}
    data = []
    with open('C:/Users/sxy/Desktop/My/Predict_dataset/exp2-36/train/Khaos.txt', 'r') as f:
        line = f.readline()
        num = 0
        while line:
            if line[:len(line) - 1] in dict:
                num += 1
                print(line, end="")
            else:
                data.append(line[:len(line) - 1])
                dict[line[:len(line) - 1]] = 1
            line = f.readline()
    f.close()
    writeToText(data, 'C:/Users/sxy/Desktop/My/Predict_dataset/exp2-36/train/Khaos-10000.txt')
    print(len(dict))
    print('repeat num: ', num)


def mkdir():
    model_name = ['RF', 'LSTM', 'LSTM_MI', 'CNN', 'CNN_LSTM', 'biLSTM']
    for name in model_name:
        os.mkdir('C:/Users/sxy/Desktop/My/Result/ROC_AUC/experiment1/' + name + '_real')


def statics_len_domain(filein):
    with open(filein, 'r') as f:
        line = f.readline()
        list = []
        while line[:len(line) - 1]:
            list.append(len(line)-1)
            line = f.readline()
    f.close()
    set1 = set(list)
    dict = {}
    for item in set1:
        dict.update({item: list.count(item)})
    print(dict)


def len_check():
    num = 0
    domain_filter = []
    with open('C:/Users/sxy/Desktop/My/Predict_dataset/exp2-36/train/Khaos-10000.txt', 'r') as f:
        line = f.readline()
        while line:
            if len(line) > 3:
                domain_filter.append(line[:len(line) - 1])
                num += 1
            line = f.readline()
    writeToText(domain_filter, 'C:/Users/sxy/Desktop/My/Predict_dataset/exp2-36/train/Khaos.txt')
    print('no resign domain have %d ' % num)


def get_csv():
    file_label = []
    for filename in glob.glob('./DGArchive/*.csv'):
        file_label.append(filename)
    print('file_label', file_label)
    data_count = []
    for name in file_label:
        table = pd.read_csv(name, header=None)
        out = name.split('/')[-1]
        data = []
        with open('dgarchive/' + out + '.txt', 'w') as f:
            for i in table[0]:
                data.append(i)
                f.write(i + '\n')
            data_count.append(len(data))
    with open('dgarchive/datacount.txt', 'w') as f:
        for k, v in zip(file_label, data_count):
            f.write(k.split('/')[-1] + ' ' + str(v) + '\n')


def get_all():
    file_label = []
    for filename in glob.glob('/home/tank05/Desktop/ana/DGA-dataset/DGArchive/*.csv'):
        file_label.append(filename)
    print('file_label', file_label)
    for name in file_label:
        table = pd.read_csv(name, header=None)
        with open('dataset/dgarchive.txt', 'a') as f:
            for i in table[0]:
                f.write(i + '\n')


def get_random_csv(filein, fileout):
    f_out = open(fileout, 'w', encoding='utf-8')
    if len(filein) > 1:
        for name in filein:
            print(name)
            f_in = open(name, 'r', encoding='utf-8')
            lines = f_in.readlines()
            if len(lines) < 1300:
                for i in lines:
                    f_out.write(i)
            else:
                resultlist = random.sample(range(0, len(lines)), len(lines)//1300)
                for i in resultlist:
                    f_out.write(lines[i])
            f_in.close()
    else:
        table = pd.read_csv(filein[0], header=None)
        data = []
        for i in table[1]:
            print(i)
            data.append(i)
        resultlist = random.sample(range(0, len(data)), 10000)
        for i in resultlist:
            f_out.write(data[i][1:] + '\n')
    f_out.close()


def list_to_txt(data, f):
    for i in data:
        f.write(str(i) + ' ')
    f.write('\n')


def all_roc(name, fpr, tpr, roc_auc, dga, label_true, label_test):
    Font = {'size': 15, 'family': 'Times New Roman'}
    Font_l = {'size': 10, 'family': 'Times New Roman'}
    plt.figure(figsize=(6, 6))
    f = open('C:/Users/sxy/Desktop/4.29/Result/Data-process/exp3-100000-' + str(name) + '.txt', 'a')  # exp1
    # f = open('C:/Users/sxy/Desktop/4.29/Result/Data-process/exp2-' + str(name) + '.txt', 'a')  # exp2
    f.write(name + '\n')
    for fpr_i, tpr_i, i_roc, dga_name, label_true_i, label_test_i in zip(fpr, tpr, roc_auc, dga, label_true, label_test):
        label = dga_name + '=%0.4f' % i_roc
        label1 = dga_name + '=%0.8f' % i_roc
        if dga_name is 'SeqGAN':
            plt.plot(fpr_i, tpr_i, label=label, linewidth=2.0)
        else:
            plt.plot(fpr_i, tpr_i, label=label)
        f.write(label1 + '\n')
        f.write('fpr ' + '\n')
        list_to_txt(fpr_i, f)
        f.write('tpr ' + '\n')
        list_to_txt(tpr_i, f)
        f.write('true label ' + '\n')
        list_to_txt(label_true_i, f)
        f.write('test label ' + '\n')
        list_to_txt(label_test_i, f)
    f.write('\n')
    plt.legend(loc='lower right', prop=Font_l)
    plt.title(name)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate', Font)
    plt.ylabel('True Positive Rate', Font)
    plt.tick_params(labelsize=10)
    plt.savefig('C:/Users/sxy/Desktop/4.29/Result/' + name + '.png', dpi=75)
    plt.show()


def all_classifier_roc(name, fpr, tpr, roc_auc):
    Font = {'size': 15, 'family': 'Times New Roman'}
    Font_l = {'size': 10, 'family': 'Times New Roman'}
    plt.figure(figsize=(6, 6))
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for fpr_i, tpr_i, i_roc, classifier_name in zip(fpr, tpr, roc_auc, name):
        label = classifier_name + '=%0.3f' % i_roc
        plt.plot(fpr_i, tpr_i, label=label)
    plt.legend(loc='lower right', prop=Font_l)
    plt.title(name)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate', Font)
    plt.ylabel('True Positive Rate', Font)
    plt.tick_params(labelsize=10)
    plt.savefig('./png/Classifier_sense/classifier.png', dpi=75)
    plt.show()


def roc(classifier, dganame, fpr, tpr, roc_auc):
    Font = {'size': 15, 'family': 'Times New Roman'}
    Font_l = {'size': 10, 'family': 'Times New Roman'}
    plt.figure(figsize=(6, 6))
    plt.legend(loc='lower right', prop=Font_l)
    plt.plot(fpr, tpr)
    # title = classifier + '-' + dganame + '=%0.3f' % roc_auc
    # plt.title(title)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    # plt.xlabel('False Positive Rate', Font)
    # plt.ylabel('True Positive Rate', Font)
    plt.tick_params(labelsize=10)
    plt.savefig('C:/Users/sxy/Desktop/My/Result/ROC_AUC/experiment1/real/' + classifier + '_real' + '/' + classifier
                + '-' + dganame + '.png', dpi=75)
    plt.show()


def split_train_test(filein, train_name, test_name, num):
    test = []
    train = []
    with open(filein, 'r') as f:
        train1 = f.readlines()
    randomresult = random.sample(range(0, len(train1)), num)
    for i in randomresult:
        test.append(train1[i])
    for i in range(0, len(train1)):
        if i not in randomresult:
            train.append(train1[i])
    with open(train_name, 'w') as f_train:
        for i in train:
            f_train.write(i)
    f_train.close()
    with open(test_name, 'w') as f_test:
        for i in test:
            f_test.write(i)
    f_test.close()


if __name__ == '__main__':
    split_train_test('C:/Users/sxy/Desktop/Dataset/exp2/SeqGAN.txt',
                     'C:/Users/sxy/Desktop/Dataset/exp2/train/SeqGAN.txt',
                     'C:/Users/sxy/Desktop/Dataset/exp2/test/SeqGAN.txt', 1000)
