import RF.classifiers as c
import sklearn.metrics as metrics
import file_check as file_check
import random
import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import KFold  # 数据划分训练集与测试集

train_benign = 128926
train_malicious = 128926
# train_benign = 101
# train_malicious = 101
test_benign = 1000
test_malicious = 1000

benign_txt = "C:/Users/sxy/Desktop/experiment/dataset/train/benign.txt"
malicious_txt = 'C:/Users/sxy/Desktop/experiment/dataset/train/malicious.txt'
# malicious_txt = './100000_m.txt'


def RF():
    benign = []
    malicious = []
    benign_b = file_check.get_no_dot(benign_txt)[:train_benign]
    malicious_b = file_check.get_no_dot(malicious_txt)[:train_malicious]

    if len(benign_b) > train_benign:
        resultlist = random.sample(range(0, len(benign_b)), train_benign)
        for i in resultlist:
            benign.append(benign_b[i])
    else:
        benign = benign_b
    if len(malicious_b) > train_malicious:
        resultlist = random.sample(range(0, len(malicious_b)), train_malicious)
        for i in resultlist:
            malicious.append(malicious_b[i])
    else:
        malicious = malicious_b
    random.shuffle(benign)
    random.shuffle(malicious)
    data = benign + malicious
    label = [0] * len(benign) + [1] * len(malicious)
    print("pre_training")
    data = np.array(data)
    label = np.array(label)
    auc_c = []
    index = [i for i in range(len(data))]
    random.shuffle(index)
    data = data[index]
    label = label[index]
    k = 5
    clf = c.RFClassifier(dga='mix', n_estimators=100, criterion='gini', max_features='auto', max_depth=50,
                         min_samples_split=2)
    auc = 0.0
    kf = KFold(n_splits=k)
    for train_index, test_index in kf.split(data):
        train_x, train_y = data[train_index], label[train_index]
        test_x, test_y = data[test_index], label[test_index]
        print(train_y, test_y)
        clf.training(train_x, train_y)
        prob = clf.predict_proba(test_x)
        print(test_y)
        t_auc = metrics.roc_auc_score(test_y, prob[:, 1])
    auc = auc/k
    # print(i, auc)
    auc_c.append(auc)
    print(auc_c)
    # 保存模型
    # joblib.dump(clf, 'RF.m')
    # 加载模型
    # clf = joblib.load('RF.m')
    # dga = ['gozi', 'kraken_v1', 'matsnu', 'pykspa', 'suppobox']
    # correct = []
    # auc = []
    # for i in dga:
    #     file = 'C:/Users/sxy/Desktop/4.29/Dataset/test/' + str(i) + '.txt'
    #     data = file_check.get_has_dot(file)
    #     label = [1 for _ in range(len(data))]
    #     t_probs = clf.predict(data)
    #     t_prob = clf.predict_proba(data)
    #     corr = 0
    #     for i in t_probs:
    #         if i == 1:
    #             corr += 1
    #     print('corr', corr)


def exp1():
    benign = []
    malicious = []
    benign_b = file_check.get_no_dot(benign_txt)[:train_benign]
    malicious_b = file_check.get_no_dot(malicious_txt)[:train_malicious]

    if len(benign_b) > train_benign:
        resultlist = random.sample(range(0, len(benign_b)), train_benign)
        for i in resultlist:
            benign.append(benign_b[i])
    else:
        benign = benign_b
    if len(malicious_b) > train_malicious:
        resultlist = random.sample(range(0, len(malicious_b)), train_malicious)
        for i in resultlist:
            malicious.append(malicious_b[i])
    else:
        malicious = malicious_b
    random.shuffle(benign)
    random.shuffle(malicious)
    data = benign + malicious
    label = [0] * len(benign) + [1] * len(malicious)
    print("pre_training")
    data = np.array(data)
    label = np.array(label)
    auc_c = []
    dis = c.RFClassifier(dga='mix', n_estimators=100, criterion='gini', max_features='auto', max_depth=50,
                         min_samples_split=2)
    dis.training(data, label)
    dga = ['our', 'khaos_original', 'kraken', 'gozi', 'suppobox']
    for i in dga:
        f = open('C:/Users/sxy/Desktop/experiment/dataset/test/benign-test.txt', 'r')
        benign = f.read().splitlines()
        f.close()
        if i == 'our':
            f = open('C:/Users/sxy/Desktop/experiment/dataset/our/exp1-CNN-test.txt')
        elif i == 'khaos_original':
            # f = open('C:/Users/sxy/Desktop/experiment/dataset/khaos_lstm_110000.txt')
            f = open('C:/Users/sxy/Desktop/experiment/dataset/test/' + str(i) + '-test.txt')
        else:
            f = open('C:/Users/sxy/Desktop/experiment/dataset/test/' + str(i) + '-test.txt')
        malicious = f.read().splitlines()
        f.close()
        index = [j for j in range(10000)]
        random.shuffle(index)
        benign_test1 = []
        malicious_test1 = []
        for k in index:
            benign_test1.append(benign[k])
            malicious_test1.append(malicious[k])
        domain = benign_test1 + malicious_test1
        y_true = [0] * len(benign_test1) + [1] * len(malicious_test1)

        prob = dis.predict_proba(domain)
        # fpr, tpr, _ = metrics.roc_curve(test_y, prob[:, 1], pos_label=1)
        fpr, tpr, _ = metrics.roc_curve(y_true, prob[:, 1], pos_label=1)
        # auc = metrics.roc_auc_score(test_y, prob[:, 1])
        auc = metrics.roc_auc_score(y_true, prob[:, 1])
        print(i, auc)


exp1()
# RF()