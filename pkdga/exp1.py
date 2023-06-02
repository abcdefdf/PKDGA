import RF.classifiers as c
import lstm.lstm_class as lstm
import textcnn.cnn as cnn
import sklearn.metrics as metrics
from statics.statics import statics_classifier
from graph.WordGraph import Wordgraph
import random
import bilstm.bilstm_class as bilstm
import pandas as pd
import lstm.khaos_lstm as khaoslstm
import textcnn.khaoscnn as khaoscnn
from openpyxl import Workbook

benign_txt = './dataset/train/benign.txt'
malicious_txt = './dataset/train/malicious.txt'


def get_txt(name):
    f = open(name, 'r')
    data = f.read().splitlines()
    f.close()
    return data


def get_data(use, dga, exp_num, model):
    benign, malicious = [], []
    if use is 'train':
        if exp_num is 1:
            malicious = get_txt(malicious_txt)
            benign = get_txt(benign_txt)
        elif exp_num is 2:
            if dga is 'our':
                # file = 'C:/Users/sxy/Desktop/experiment/dataset/our/exp1-' + str(model) + '-train.txt'
                file = 'C:/Users/sxy/Desktop/experiment/dataset/our/2/exp1-LSTM-train.txt'
            else:
                file = 'C:/Users/sxy/Desktop/experiment/dataset/statics/' + str(dga) + '-train.txt'
            benign_original = get_txt('C:/Users/sxy/Desktop/experiment/dataset/statics/benign-train.txt')
            # file = 'C:/Users/sxy/Desktop/experiment/dataset/statics/' + str(dga) + '-train.txt'
            malicious_original = get_txt(file)
            trainnum = 100000
            if len(malicious_original) > trainnum:
                result_benign = random.sample(range(0, len(benign_original)), trainnum)
                result_malicious = random.sample(range(0, len(malicious_original)), trainnum)
                malicious = []
                benign = []
                for i in result_benign:
                    benign.append(benign_original[i])
                for i in result_malicious:
                    malicious.append(malicious_original[i])
            else:
                result_benign = random.sample(range(0, len(benign_original)), len(malicious_original))
                benign = []
                for i in result_benign:
                    benign.append(benign_original[i])
                malicious = malicious_original
    elif use is 'test':
        if dga is 'our':
            # file = 'C:/Users/sxy/Desktop/experiment/dataset/our/1/exp1-' + str(model) + '-test.txt'
            file = 'C:/Users/sxy/Desktop/experiment/dataset/our/2/exp1-LSTM-test.txt'
        elif dga is 'maskDGA':
            file = 'C:/Users/sxy/Desktop/experiment/dataset/statics/maskDGA-train.txt'
        else:
            file = 'C:/Users/sxy/Desktop/experiment/dataset/test/' + str(dga) + '-test.txt'
        test_benign_txt = 'C:/Users/sxy/Desktop/experiment/dataset/test/benign-test.txt'
        malicious_all = get_txt(file)
        benign_all = get_txt(test_benign_txt)
        # if dga is 'charbot':
        #     result_benign = random.sample(range(0, len(benign_all)), 2000)
        #     result_malicious = random.sample(range(0, len(malicious_all)), 2000)
        if dga is 'maskDGA':
            result_benign = random.sample(range(0, len(benign_all)), 1000)
            result_malicious = random.sample(range(0, len(malicious_all)), 1000)
        # elif dga is 'maskDGA':
        #     result_benign = random.sample(range(0, len(benign_all)), 1000)
        #     result_malicious = random.sample(range(0, len(malicious_all)), 1000)
        else:
            result_benign = random.sample(range(0, len(benign_all)), 10000)
            result_malicious = random.sample(range(0, len(malicious_all)), 10000)
        malicious = []
        benign = []
        for i in result_benign:
            benign.append(benign_all[i])
        for i in result_malicious:
            malicious.append(malicious_all[i])
        # file = './dataset/test/' + str(dga) + '-test.txt'
        # test_benign_txt = './dataset/test/benign-test.txt'
        # malicious = get_txt(file)
        # benign = get_txt(test_benign_txt)
    elif use is 'statics':
        if exp_num is 2:
            malicious = get_txt('./dataset/statics/' + str(dga) + '-train.txt')
        else:
            malicious = get_txt(malicious_txt)
        benign = get_txt('./dataset/statics/benign-train.txt')
        # if exp_num is 1:
        #     malicious = get_txt(malicious_txt)
        #     benign = get_txt(benign_txt)
        # else:
        #     malicious = get_txt('./dataset/train/exp2/' + str(dga) + '-train.txt')
        #     benign = get_txt('./dataset/train/exp2/benign-train.txt')
        return benign, malicious
    label = [0] * len(benign) + [1] * len(malicious)
    data = benign + malicious
    return data, label


def write_txt(dga, dga_train, auc, prob, test_y, file, name):
    f = open(file, 'a')
    f.write(str(name) + '-' + str(dga_train) + '-' + str(dga))
    f.write('auc = ' + str(auc) + '\n')
    for i in prob:
        f.write(str(i) + ' ')
    f.write('\n')
    if name is 'LSTM' or name is 'CNN':
        # for i in range(len(test_y)):
        #     test_y[i] = test_y[i]
        for i in test_y:
            f.write(str(i.item()) + ' ')
    else:
        for i in test_y:
            f.write(str(i) + ' ')
    f.write('\n')

    # workbook = Workbook()
    # save_file = 'C:/Users/sxy/Desktop/experiment/Result/new/exp1_'+str(dga)+'.xlsx'
    # worksheet = workbook.active
    # # 每个workbook创建后，默认会存在一个worksheet，对默认的worksheet进行重命名
    # worksheet.title = "Sheet1"
    # l = [i for i in range(20000)]
    # worksheet.append(l)
    # worksheet.append(list(prob))  # 把每一行append到worksheet中
    # worksheet.append(list(test_y))  # 把每一行append到worksheet中
    # workbook.save(filename=save_file)  # 不能忘
    #
    # test = pd.DataFrame({'1': test_y, '2': prob})
    # test.to_csv('C:/Users/sxy/Desktop/experiment/Result/new/exp1_'+str(name)+'.csv')
    # f1 = open('C:/Users/sxy/Desktop/experiment/Result/new/exp1_'+str(dga)+'.csv', 'a', newline='')
    #
    # # plt.title('%s auc = %f' % (i, t_auc))
    # # if t_auc < 0.5:
    # #     prob = []
    # #     for i in proba:
    # #         prob.append(1/i)
    # #     proba = prob
    # #     t_auc = metrics.roc_auc_score(y_true, proba)
    # writer = csv.writer(f1)
    # writer.writerows(test_y)
    # writer.writerows(prob)
    # f1.close()
    f.close()


def classifier(name, exp_num, dga, cuda, dis_num):
    train_x, train_y = [], []
    if name is'Statics':
        train_x, train_y = get_data('Statics', dga, 1, '')
    elif exp_num is 1:
        train_x, train_y = get_data('train', 'no', 1, '')
    elif exp_num is 2:
        train_x, train_y = get_data('train', dga, 2, name)
    dis = ''
    if name is 'RF':
        dis = c.RFClassifier(dga='mix', n_estimators=100, criterion='gini', max_features='auto', max_depth=50,
                             min_samples_split=2)
        dis.training(train_x, train_y)
    elif name is 'LSTM':
        dis = lstm.training(train_x, train_y, dis_num)
    elif name is 'CNN':
        dis = cnn.training(train_x, train_y, cuda, dis_num)
    elif name is 'bilstm':
        dis = bilstm.train(train_x, train_y, dis_num)
    elif name is 'Statics':
        benign, malicious = get_data('statics', 'no', 1, '')
        # benign, malicious = get_data('statics', dga, 2, '')
        dis = statics_classifier(benign, malicious)
    elif name is 'Graph':
        f = open('C:/Users/sxy/Desktop/experiment/dataset/graph/exp1-graph-dict.txt', 'r')
        data_dic = f.read().splitlines()
        f.close()
        dictionary = data_dic[1:]
        label = data_dic[0]
        label = int(label)
        dis = Wordgraph(dictionary, label)
    elif name is 'khaoslstm':
        dis = khaoslstm.training(train_x, train_y)
    elif name is 'khaoscnn':
        dis = khaoscnn.training(train_x, train_y)
    return dis


def test_classifier():
    name = "LSTM"
    dis = classifier(name, 1, 'no', cuda=False)
    dga = ['khaos_lstm', 'khaos_original', 'kraken', 'charbot', 'gozi', 'suppobox', 'pykspa', 'our']
    for i in dga:
        test_x, test_y = get_data('test', i, 1, name)
        test_x, test_y = lstm.get_data(test_x, test_y)
        prob = lstm.predict_proba(dis, test_x)
        prob = prob.detach().numpy()
        fpr, tpr, _ = metrics.roc_curve(test_y, prob[:, 1], pos_label=1)
        auc = metrics.roc_auc_score(test_y, prob[:, 1])
        print(name, i, auc)


def classification(name, exp_num, dga_train, dga, cuda):
    file = './Result/new/exp2-' + str(name) + '.txt'
    # dga = ['khaos_lstm']
    # dga = ['maskDGA']
    # dga = ['khaos_original', 'kraken', 'gozi', 'suppobox', 'maskDGA', 'our']
    for i in range(1):
        for dga_train_i in dga_train:
            dis = classifier(name, exp_num, dga_train_i, cuda, 10)
            # print(name, i)
            # dis = classifier(name, exp_num, dga_train_i)
            for dga_i in dga:
                # dis = classifier(name, exp_num, dga_i)
                print('train', dga_train_i, 'test', dga_i)
                test_x, test_y = get_data('test', dga_i, 1, name)
                # print(len(test_x), test_x[0])
                if name is 'RF':
                    prob = dis.predict_proba(test_x)
                    # fpr, tpr, _ = metrics.roc_curve(test_y, prob[:, 1], pos_label=1)
                    fpr, tpr, _ = metrics.roc_curve(test_y, prob[:, 0], pos_label=1)
                    # auc = metrics.roc_auc_score(test_y, prob[:, 1])
                    auc = metrics.roc_auc_score(test_y, prob[:, 1])
                    print(dga_i, auc)
                    write_txt(dga_i, dga_train_i, auc, prob[:, 1], test_y, file, name)
                elif name is 'LSTM':
                    test_x, test_y = lstm.get_data(test_x, test_y)
                    prob = lstm.predict_proba(dis, test_x)
                    prob = prob.detach().numpy()
                    fpr, tpr, _ = metrics.roc_curve(test_y, prob[:, 1], pos_label=1)
                    auc = metrics.roc_auc_score(test_y, prob[:, 1])
                    print(dga_i, auc)
                    write_txt(dga_i, dga_train_i, auc, prob[:, 0], test_y, file, name)
                elif name is 'CNN':
                    test_x, test_y = cnn.get_data(test_x, test_y)
                    prob = cnn.predict_proba(dis, test_x)
                    prob = prob.detach().numpy()
                    fpr, tpr, _ = metrics.roc_curve(test_y, prob[:, 1], pos_label=1)
                    auc = metrics.roc_auc_score(test_y, prob[:, 1])
                    print(dga_i, auc)
                    write_txt(dga_i, dga_train_i, auc, prob[:, 0], test_y, file, name)
                elif name is 'bilstm':
                    test_x, test_y = lstm.get_data(test_x, test_y)
                    prob = bilstm.predict(dis, test_x)
                    fpr, tpr, _ = metrics.roc_curve(test_y, prob, pos_label=1)
                    auc = metrics.roc_auc_score(test_y, prob)
                    print(dga_i, auc)
                    write_txt(dga_i, dga_train_i, auc, prob[:, 0], test_y, file, name)
                elif name is 'Statics':
                    prob, _ = dis.predict_proba(test_x, 1000)
                    auc = metrics.roc_auc_score(test_y, prob)
                    fpr, tpr, _ = metrics.roc_curve(test_y, prob, pos_label=1)
                    write_txt(dga_i, dga_train_i, auc, prob[:, 0], test_y, file, name)
                    print(prob)
                    print(auc)
                elif name is 'Graph':
                    prob = dis.predict_proba(test_x)
                    auc = metrics.roc_auc_score(test_y, prob)
                    fpr, tpr, _ = metrics.roc_curve(test_y, prob, pos_label=1)
                    write_txt(dga_i, dga_train_i, auc, prob[:, 0], test_y, file, name)
                    print(prob)
                    print(auc)
                elif name is 'khaoslstm':
                    print('gg')
                    test_x, test_y = khaoslstm.get_data(test_x, test_y)
                    prob = khaoslstm.predict_proba(dis, test_x)
                    prob = prob.detach().numpy()
                    fpr, tpr, _ = metrics.roc_curve(test_y, prob[:, 1], pos_label=1)
                    auc = metrics.roc_auc_score(test_y, prob[:, 1])
                    print('auc %f' % auc)
                    write_txt(dga_i, dga_train_i, auc, prob[:, 0], test_y, file, name)
                elif name is 'khaoscnn':
                    test_x, test_y = khaoscnn.get_data(test_x, test_y)
                    prob = khaoscnn.predict_proba(dis, test_x)
                    prob = prob.detach().numpy()
                    fpr, tpr, _ = metrics.roc_curve(test_y, prob[:, 1], pos_label=1)
                    auc = metrics.roc_auc_score(test_y, prob[:, 1])
                    write_txt(dga_i, dga_train_i, auc, prob[:, 0], test_y, file, name)


def classification_1(name, dga_train, cuda):
    print(dga_train)
    dis = classifier(name, 2, dga_train, cuda)
    test_x, test_y = get_data('test', dga_train, 2, '')
    if name is 'Statics':
        print('JI')
        prob_ji, proba = dis.predict_proba(test_x)
        auc_ji = metrics.roc_auc_score(test_y, prob_ji)
        print(prob_ji)
        print(auc_ji)

        print('ED')
        prob_ed = dis.ED_our(test_x)
        auc_ed = metrics.roc_auc_score(test_y, prob_ed)
        print(prob_ed)
        print(auc_ed)

        print('KL')
        prob_kl = dis.KL_Our(test_x)
        auc_kl = metrics.roc_auc_score(test_y, prob_kl)
        print(prob_kl)
        print(auc_kl)
    elif name is 'Graph':
        prob = dis.predict_proba(test_x)
        auc = metrics.roc_auc_score(test_y, prob)
        print(prob)
        print(auc)


def get_auc(dis):
    test_x, test_y = get_data('test', 'khaos_lstm', 1, '')
    test_x, test_y = lstm.get_data(test_x, test_y)
    pred = dis(test_x)
    auc = metrics.roc_auc_score(test_y, pred)
    print(auc)


if __name__ == '__main__':
    # test_classifier()
    name = ['LSTM', 'CNN']
    # name = ['RF']
    cuda = False
    # dga = ['kraken', 'charbot', 'gozi', 'suppobox', 'pykspa']
    dga_test = ['maskDGA']
    dga = ['khaos_original', 'kraken', 'gozi', 'suppobox', 'maskDGA', 'our']
    # dga = ['khaos_lstm', 'khaos_original']
    # dga = ['maskDGA']
    for model in name:
        classification(model, 2, dga, dga_test, cuda)
    # classification('Graph', 1, ['no'])
