import networkx as nx
import csv
import matplotlib.pyplot as plt
from sklearn import tree  # 导包
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import random
from tqdm import tqdm
import sklearn.metrics as metrics


# 获取数据
def get_data(name):
    f = open(name, 'r')
    data = f.read().splitlines()
    f.close()
    train_x = []
    train_y = []
    for i in data:
        i = i.replace(' ', '')
        c = i.split(',')
        x = np.array(c[:5])
        x = x.astype(np.float)
        y = np.array(int(c[5]))
        train_x.append(x)
        train_y.append(y)
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    return train_x, train_y


def getNumofCommonSubstr(str1, str2):
    """
    求两个字符串的最长公共子串
    思想：建立一个二维数组，保存连续位相同与否的状态
    """
    lstr1 = len(str1)
    lstr2 = len(str2)
    record = [[0 for i in range(lstr2 + 1)] for j in range(lstr1 + 1)]  # 多一位
    maxNum = 0  # 最长匹配长度
    p = 0  # 匹配的起始位

    for i in range(lstr1):
        for j in range(lstr2):
            if str1[i] == str2[j]:
                # 相同则累加
                record[i + 1][j + 1] = record[i][j] + 1
                if record[i + 1][j + 1] > maxNum:
                    # 获取最大匹配长度
                    maxNum = record[i + 1][j + 1]
                    # 记录最大匹配长度的终止位置
                    p = i + 1
    return str1[p - maxNum:p], maxNum


def str_int(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    length = len(s1)
    result = []
    for step in range(length, 0, -1):
        for start in range(0, length-step+1):
            flag = True
            tmp = s1[start:start+step]
            if s2.find(tmp) > -1:  # 第一次找到,后面要接着找
                result.append(tmp)
                flag = True
                newstart = start+1
                newstep = step
                while flag:   # 已经找到最长子串,接下来就是判断后面是否还有相同长度的字符串
                    if newstart + step > length:  # 大于字符串总长了,退出循环
                        break
                    newtmp = s1[newstart:newstart+newstep]
                    if s2.find(newtmp) > -1:
                        result.append(newtmp)
                        newstart += 1
                        flag = True
                    else:
                        newstart += 1
                        flag = True
                return result
            else:
                continue


def split_str_forward(domain, begin, sub, dect):
    flag = False
    ad = 3
    temp = begin
    while flag is False and temp < len(domain)-3:
        begin = temp
        while begin+ad < len(domain)+1:
            a = domain[begin:begin + ad]
            if domain[begin:begin + ad] in dect:
                if begin+ad is len(domain) or domain[begin:begin + ad + 1] not in dect:
                    sub.append(domain[begin:begin + ad])
                    begin = begin + ad
                    flag = True
                    break
            ad += 1
        if flag is True:
            return sub, begin
        else:
            ad = 3
            temp += 1
    return sub, 'no'


def split_str_backward(domain, end, sub, dect):
    flag = False
    ad = 3
    temp = end
    while flag is False and temp > 2:
        end = temp
        while end-ad+1 > 0:
            a = domain[end-ad:end]
            if domain[end-ad:end] in dect:
                # if domain[end-ad-1:end] not in dect or end is 0:
                sub.append(domain[end-ad:end])
                end = end - ad
                flag = True
                break
            ad += 1
        if flag is True:
            return sub, end
        else:
            ad = 3
            temp -= 1
    return sub, 'no'


def get_sub(dect, domain):
    sub_forward = []
    sub_backward = []
    sub = []
    begin = 0
    end = len(domain)
    while begin is not 'no' and begin is not len(domain)-1:
        sub_forward, begin = split_str_forward(domain, begin, sub_forward, dect)
    while end is not 'no' and end is not 0:
        sub_backward, end = split_str_backward(domain, end, sub_backward, dect)
    sub_forward = sorted(sub_forward)
    sub_backward = sorted(sub_backward)
    if sub_forward == sub_backward:
        sub = sub_forward
    else:
        if len(sub_forward) > len(sub_backward):
            sub = sub_backward
        elif len(sub_forward) < len(sub_backward):
            sub = sub_forward
        else:
            forward = ''.join(sub_forward)
            backward = ''.join(sub_backward)
            if len(forward) > len(backward):
                sub = sub_forward
            elif len(forward) < len(backward):
                sub = sub_backward
            else:
                sub = sub_forward
    return sub


def gozi(name):
    f = open('C:/Users/sxy/Desktop/experiment/dataset/test/maskDGA-test.txt', 'r')
    data = f.read().splitlines()
    # data = []
    # for i in d:
    #     s = i.split('.')
    #     data.append(s[0])
    data = list(set(data))
    f.close()
    feature, dictionary = create_graph(data)
    f = open('C:/Users/sxy/Desktop/experiment/dataset/graph/maskDGA-dict.txt', 'a')
    for i in dictionary[0]:
        f.write(str(i) + '\n')
    print(feature, dictionary)
    # result = random.sample(range(0, len(data)), 128926)
    # f = open('./benign', 'a')
    # for i in range(0, len(result)):
    #     f.write(data[result[i]] + '\n')
    # f.close()

    # f = open('./' + str(name) + '_dga.txt', 'r')
    # data = f.read().splitlines()
    # f.close()
    #
    # f = open('./' + str(name) + '_train.txt', 'a')
    # f_test = open('./' + str(name) + '_test.txt', 'a')
    # result = random.sample(range(0, len(data)), 11000)
    #
    # for i in range(0, 10000):
    #     f.write(data[result[i]] + '\n')
    # f.close()
    # for i in range(10000, 11000):
    #     f_test.write(data[result[i]] + '\n')
    # f_test.close()


def create_graph(data):
    feature = []
    dictionary = []
    # 建字典
    dect = []
    for i in tqdm(range(0, len(data))):
    # for i in tqdm(range(0, 100)):
        j = i + 1
        while j < len(data):
            comstr, _ = getNumofCommonSubstr(data[i], data[j])
            # comstr = str_int(d[i], d[j])
            if comstr != None and len(comstr) > 2:
                dect.append(comstr)
            j += 1
    dect = list(set(dect))
    # 创建图
    g = nx.Graph()  # 创建空的无向图
    for i in dect:
        g.add_node(i)
    for i in tqdm(data):
        sub = get_sub(dect, i)
        if len(sub) > 1:
            for j in range(0, len(sub)):
                k = j + 1
                while k < len(sub):
                    g.add_edge(sub[j], sub[k])
                    k += 1
    # 删除节点
    remove_nodes = []
    for i in g.nodes():
        # print(i)
        if g.degree(i) < 4:
            remove_nodes.append(i)
    for i in remove_nodes:
        g.remove_node(i)
    for c in nx.connected_components(g):
        subgraph = g.subgraph(c)
        if len(subgraph.nodes()) > 100:
            sub_dictionary = subgraph.nodes()
            degree_ave = 0
            degree_max = 0
            subgraph = g.subgraph(c)
            if len(subgraph.nodes()) > 1:
                for i in subgraph.degree:
                    degree_ave += i[1]
                    if i[1] > degree_max:
                        degree_max = i[1]
                degree_ave = degree_ave / len(subgraph.degree)
            C = nx.cycle_basis(subgraph)
            feature_i = [degree_ave, degree_max, len(C), len(C) / len(g.nodes()),
                         nx.average_shortest_path_length(subgraph)]
            feature.append(feature_i)
            dictionary.append(list(sub_dictionary))
    return feature, dictionary


# def predict(data):
#     train_x, train_y = get_data('./graph.txt')
#     # 实例化模型，划分规则用的是熵
#     clf = tree.DecisionTreeClassifier(criterion="gini")
#
#     # 拟合数据
#     clf = clf.fit(train_x, train_y)
#     feature, dictionary = create_graph(data)
#     predict = []
#     for feature_i, dictionary_i in zip(feature, dictionary):
#         # predict_i = []
#         prob = clf.predict_proba([feature_i])
#         for domain_i in data:
#             # print(dictionary_i, domain_i)
#             sub = get_sub(dictionary_i, domain_i)
#             if len(sub) > 1:
#                 predict.append(prob[0][0])
#             else:
#                 predict.append(prob[0][1])
#     # predict.append(predict_i)
#     return predict


class Wordgraph:
    def __init__(self, dictionary, label):
        self.dictionary = dictionary
        self.label = label

    def predict_proba(self, data):
        # f = open('C:/Users/sxy/Desktop/experiment/dataset/graph/exp1-graph-dict.txt', 'r')
        # data_dic = f.read().splitlines()
        # f.close()
        # dictionary_file = data_dic[1:]
        # label = data_dic[0]
        # label = int(label)
        predict = []
        prob = []
        for domain_i in data:
            # print(dictionary_i, domain_i)
            sub = get_sub(self.dictionary, domain_i)
            if len(sub) > 1:
                predict.append(self.label)
            else:
                predict.append(1 - self.label)
            prob.append(len(sub))
        return predict, prob

        # train_x, train_y = get_data('C:/Users/sxy/Desktop/experiment/graph/graph.txt')
        # # 实例化模型，划分规则用的是熵
        # clf = tree.DecisionTreeClassifier(criterion="entropy")
        #
        # # clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
        # # 拟合数据
        # clf = clf.fit(train_x, train_y)
        # feature, dictionary = create_graph(data)
        # print(feature)
        # predict = []
        # for feature_i, dictionary_i in zip(feature, dictionary):
        #     # predict_i = []
        #     prob = clf.predict([feature_i])
        #     print('prob', prob)
        #     for domain_i in data:
        #         # print(dictionary_i, domain_i)
        #         sub = get_sub(dictionary_i, domain_i)
        #         if len(sub) > 1:
        #             predict.append(prob[0])
        #         else:
        #             predict.append(1-prob[0])
        # # predict.append(predict_i)
        # return predict


def write_txt(dga_train, dga_test, auc, fpr, tpr, file, y_true, predict):
    f = open(file, 'a')
    f.write(str(dga_train) + '-' + str(dga_test) + '\n')
    f.write('auc = ' + str(auc) + '\n')
    for i in y_true:
        f.write(str(i) + ' ')
    f.write('\n')
    for i in predict:
        f.write(str(i) + ' ')
    f.write('\n')
    for i in fpr:
        f.write(str(i) + ' ')
    f.write('\n')
    for i in tpr:
        f.write(str(i) + ' ')
    f.write('\n')
    f.close()


if __name__ == '__main__':
    # gozi('a')
    # dga = ['khaos_original', 'kraken', 'gozi', 'suppobox', 'maskDGA', 'our']
    # dga = ['our', 'khaos_lstm', 'khaos_original', 'kraken', 'charbot', 'gozi', 'suppobox']
    dga = ['maskDGA']
    # # 构造字典
    # for name in dga:
    #     data = []
    #     # f = open('C:/Users/sxy/Desktop/experiment/dataset/statics/' + str(name) + '-train.txt', 'r')
    #     # f = open('C:/Users/sxy/Desktop/experiment/dataset/train/malicious.txt', 'r')
    #     f = open('C:/Users/sxy/Desktop/experiment/dataset/our/exp1-CNN-train.txt')
    #     benign = f.read().splitlines()
    #     f.close()
    #     result = random.sample(range(0, len(benign)), 9999)
    #     for i in result:
    #         data.append(benign[i])
    #     # f = open('C:/Users/sxy/Desktop/experiment/dataset/statics/benign-train.txt', 'r')
    #     f = open('C:/Users/sxy/Desktop/experiment/dataset/train/benign.txt', 'r')
    #     malicious = f.read().splitlines()
    #     f.close()
    #     result = random.sample(range(0, len(malicious)), 9999)
    #     for i in result:
    #         data.append(malicious[i])
    #     feature, dictionary = create_graph(data)
    #     train_x, train_y = get_data('C:/Users/sxy/Desktop/experiment/graph/graph.txt')
    #     # 实例化模型，划分规则用的是熵
    #     clf = tree.DecisionTreeClassifier(criterion="entropy")
    #     # 拟合数据
    #     clf = clf.fit(train_x, train_y)
    #     prob = clf.predict(feature)
    #     f = open('C:/Users/sxy/Desktop/experiment/dataset/graph/exp1-CNN-our-dict.txt', 'a')
    #     # f = open('C:/Users/sxy/Desktop/experiment/dataset/graph/' + str(name) + '-dict.txt', 'a')
    #     f.write(str(prob))
    #     for i in dictionary:
    #         for j in i:
    #             f.write(j + '\n')
    #     f.close()
    #     print(dictionary)
    # dga_test = ['khaos_lstm']
    # dga_test = ['our', 'khaos_original', 'kraken', 'gozi', 'suppobox']
    # dga_test = ['kraken', 'suppobox', 'pykspa', 'gozi', 'khaos_lstm', 'khaos_original', 'charbot', 'our']
    dga_test = ['khaos_original', 'kraken', 'gozi', 'suppobox', 'maskDGA', 'our']
    # # 获取节点
    for name in dga:
        if name == 'our':
            f = open('C:/Users/sxy/Desktop/experiment/dataset/graph/exp1-CNN-our-dict.txt', 'r')
        else:
            f = open('C:/Users/sxy/Desktop/experiment/dataset/graph/' + str(name) + '-dict.txt', 'r')
        # f = open('C:/Users/sxy/Desktop/experiment/dataset/graph/exp1-graph-dict.txt', 'r')
        data = f.read().splitlines()
        f.close()
        dictionary_file = data[1:]
        label = data[0]
        label = int(label)
        f_new = open('C:/Users/sxy/Desktop/experiment/Result/new/graph_exp1_maskDGA.txt', 'a')
        for test_name in dga_test:
            f = open('C:/Users/sxy/Desktop/experiment/dataset/test/benign-test.txt', 'r')
            benign = f.read().splitlines()
            f.close()
            if test_name == 'our':
                f = open('C:/Users/sxy/Desktop/experiment/dataset/our/exp1-CNN-test.txt')
            # elif test_name == 'maskDGA' or test_name == 'khaos_original':
            #     # f = open('C:/Users/sxy/Desktop/experiment/dataset/khaos_lstm_110000.txt')
            #     f = open('C:/Users/sxy/Desktop/experiment/dataset/test/' + str(test_name) + '-test.txt')
            else:
                f = open('C:/Users/sxy/Desktop/experiment/dataset/test/' + str(test_name) + '-test.txt')
            malicious = f.read().splitlines()
            f.close()
            if test_name == 'charbot':
                index = [i for i in range(2000)]
            elif test_name == 'maskDGA':
                index = [i for i in range(1000)]
            else:
                index = [i for i in range(10000)]
            random.shuffle(index)
            benign_test1 = []
            malicious_test1 = []
            for k in index:
                benign_test1.append(benign[k])
                malicious_test1.append(malicious[k])
            domain = benign_test1 + malicious_test1
            # domain = benign + malicious
            y_true = [0] * len(benign_test1) + [1] * len(malicious_test1)
            predict = []
            prob = []
            for domain_i in tqdm(domain):
                # print(dictionary_i, domain_i)
                sub = get_sub(dictionary_file, domain_i)
                if len(sub) > 1:
                    predict.append(label)
                else:
                    predict.append(1-label)
                prob.append(len(sub))
            # auc = metrics.roc_auc_score(y_true, predict)
            # t_auc = metrics.roc_auc_score(y_true, prob_1)
            prob_1 = []
            for i in prob:
                prob_1.append(5-i)
            t_auc = metrics.roc_auc_score(y_true, prob_1)
            print('train', name, 'test', test_name, t_auc)
            f_new.write(str(name) + ' ' + str(test_name) + '\n')
            f_new.write('auc: ' + str(t_auc) + '\n')
            fpr, tpr, threshold = metrics.roc_curve(y_true, prob_1)
            for i in prob_1:
                f_new.write(str(i) + ' ')
            f_new.write('\n')

            for i in y_true:
                f_new.write(str(i) + ' ')
            # f1 = open('C:/Users/sxy/Desktop/experiment/Result/new/graph_exp1_maskDGA.csv', 'a', newline='')
            # writer = csv.writer(f1)
            # writer.writerow(prob)
            # writer.writerow(y_true)
            # writer.writerow(fpr)
            # writer.writerow(tpr)
            # plt.title('%s auc = %f' % (name, t_auc))
            # plt.plot(fpr, tpr, 'b', label='Val AUC = %0.3f' % t_auc)
            # plt.show()
            # if auc < 0.5:
            #     predict_a = []
            #     for i in predict:
            #         predict_a.append(1-i)
            #     auc = metrics.roc_auc_score(y_true, predict_a)
            #     fpr, tpr, _ = metrics.roc_curve(y_true, predict_a, pos_label=1)
            #     file = 'C:/Users/sxy/Desktop/experiment/Result/data/graph.txt'
            #     write_txt(name, test_name, auc, fpr, tpr, file, y_true, predict)
            # else:
            #     fpr, tpr, _ = metrics.roc_curve(y_true, predict, pos_label=1)
            #     file = 'C:/Users/sxy/Desktop/experiment/Result/data/graph.txt'
            #     write_txt(name, test_name, auc, fpr, tpr, file, y_true, predict)
            print(prob)
            print(t_auc)

