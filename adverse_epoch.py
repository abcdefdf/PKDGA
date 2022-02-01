import exp1
import lstm.lstm_class as lstm
import data_init
import torch
import random
import file_check
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import VAE.naomal as vae
from tqdm import tqdm
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import sequence
import torch.utils.data as Data


VOCAB_SIZE = 39
cuda = False


class VAE(nn.Module):
    def __init__(self, input_dim):
        super(VAE, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
        )
        self.fc21 = nn.Linear(16, 3)   # mean
        self.fc22 = nn.Linear(16, 3)   # var
        # self.fc3 = nn.Linear(20, 64)
        self.fc4 = nn.Sequential(
            nn.Linear(3, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()                     # 矩阵点对点相乘之后再把这些元素作为e的指数
        eps = torch.FloatTensor(std.size()).normal_()    # 生成随机数组
        # if torch.cuda.is_available():
        #     eps = eps.cuda()
        return eps.mul(std).add_(mu)    # 用一个标准正态分布乘标准差，再加上均值，使隐含向量变为正太分布

    def decode(self, z):
        # h3 = F.relu(self.fc3(z))
        # return torch.tanh(self.fc4(h3))
        return self.fc4(z)

    def forward(self, x):
        mu, logvar = self.encode(x)          # 编码
        z = self.reparametrize(mu, logvar)   # 重新参数化成正态分布
        return self.decode(z), mu, logvar    # 解码，同时输出均值方差


class LstmAutoEncoder(nn.Module):
    def __init__(self, input_layer=80, hidden_layer=100, batch_size=20):
        super(LstmAutoEncoder, self).__init__()

        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.batch_size = batch_size
        self.encoder_lstm = nn.LSTM(self.input_layer, self.hidden_layer, batch_first=True)
        self.decoder_lstm = nn.LSTM(self.hidden_layer, self.input_layer, batch_first=True)

    def forward(self, input_x):
        input_x = input_x.view(len(input_x), 1, -1)
        # encoder
        encoder_lstm, (n, c) = self.encoder_lstm(input_x,
                                                 (torch.zeros(1, self.batch_size, self.hidden_layer),
                                                  torch.zeros(1, self.batch_size, self.hidden_layer)))
        # decoder
        decoder_lstm, (n, c) = self.decoder_lstm(encoder_lstm,
                                                 (torch.zeros(1, self.batch_size, self.input_layer),
                                                  torch.zeros(1, self.batch_size, self.input_layer)))
        return decoder_lstm.squeeze()


def get_file_domain(file):
    f = open(file, 'r')
    data = f.readlines()
    for i in range(len(data)):
        data[i] = data[i][:-1]
    benign = []
    benign_b = file_check.get_no_dot(data_init.benign_txt)
    resultlist = random.sample(range(0, len(benign_b)), len(data))
    for i in resultlist:
        # data.append(benign_b[i])
        benign.append(benign_b[i])
    data_all = benign + data
    domain = [[data_init.char_to_int[y] for y in x] for x in data_all]
    domain = sequence.pad_sequences(domain, data_init.g_sequence_len)
    # label = [0] * int(len(data)/2) + [1] * int(len(data)/2)
    label = [0] * len(benign) + [1] * len(data)
    # print(domain)
    x_train, x_test, y_train, y_test = train_test_split(domain, label, test_size=0.2)
    # print(domain, label)
    x_train = torch.LongTensor(x_train)
    x_test = torch.LongTensor(x_test)
    y_test = torch.tensor(y_test)
    y_train = torch.tensor(y_train)
    # print(domain, label)
    return benign, data, data_all, label, x_train, x_test, y_train, y_test


def lstm_training(domain, label):
    domain = torch.tensor(domain)
    label = torch.tensor(label)
    dis = lstm.Discriminator(2, VOCAB_SIZE, 32, 32, cuda, 1)
    dis_criterion = nn.NLLLoss(reduction='sum')
    dis_optimizer = optim.Adam(dis.parameters())
    if cuda:
        dis = dis.cuda()
        dis_criterion = dis_criterion.cuda()
    print('Pretrain discriminator')
    index = [i for i in range(len(domain))]
    random.shuffle(index)
    domain = domain[index]
    label = label[index]
    for epoch in tqdm(range(200)):
        dis_data_iter = lstm.loader(domain, label)
        loss = lstm.train_dis_epoch(dis, dis_data_iter, dis_criterion, dis_optimizer)
        print('Epoch [%d], loss: %f' % (epoch, loss))
    return dis


def lstm_test():
    _, _, _, _, x_train, x_test, y_train, y_test = get_file_domain('adverse_samples/12_20_num1/mle.txt')
    # x_train, x_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2)
    # X = sequence.pad_sequences(X, data_init.g_sequence_len)
    # X = torch.LongTensor(X)
    # discriminator = lstm.lstm_class.Discriminator(2, len(data_init.alphabet), 32, 32, False, 1)
    # discriminator = exp1.classifier('LSTM', 1, 'no', False)
    discriminator = lstm_training(x_train, y_train)
    # torch.save(discriminator.state_dict(), 'model/lstm_100.pt')
    # discriminator.load_state_dict(torch.load('model/lstm.pt'))
    t_probs = discriminator.predict_proba(x_test)
    t_probs = t_probs.detach().numpy()
    t_auc = metrics.roc_auc_score(y_test, t_probs[:, 1])
    print(t_auc)


def VAE_test():
    f = open(data_init.benign_txt, 'r')
    benign_all = f.readlines()
    benign = benign_all[:]
    _, malicious, vae_test, label, _, _, _, _ = get_file_domain('adverse_samples/12_20_num1/0.txt')
    vae_model = vae.VAE(80)
    vae_data = vae.dataprocess(benign)
    # vae_data = vae.dataprocess(malicious)
    torch_vae = Data.TensorDataset(vae_data)
    vae_loader = Data.DataLoader(dataset=torch_vae, batch_size=128, shuffle=True)
    vae.VAE_train(vae_model, 100, vae_loader)
    #
    # vae_test = malicious_all[-128:]
    vae_test = vae.dataprocess(vae_test)
    # vae_test = Data.TensorDataset(vae_test)
    # vae_test = Data.DataLoader(dataset=vae_test, batch_size=BATCH_SIZE, shuffle=True)
    loss_f = torch.nn.MSELoss()
    prob = vae.VAE_predict(vae_test, vae_model, loss_f)
    # print(prob)
    auc = metrics.roc_auc_score(label, prob)
    print('auc', auc)


def lstm_vae():
    f = open(data_init.benign_txt, 'r')
    benign_all = f.readlines()
    benign = benign_all[:]
    _, malicious, vae_test, label, _, _, _, _ = get_file_domain('adverse_samples/12_20_num1/0.txt')
    model = LstmAutoEncoder()
    data = vae.dataprocess(benign)
    torch_vae = Data.TensorDataset(data)
    loader = Data.DataLoader(dataset=torch_vae, batch_size=20, shuffle=True)
    loss_function = nn.MSELoss()  # loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器
    epochs = 100
    # 开始训练
    model.train()
    for i in range(epochs):
        for seq in loader:
            seq = seq[0]
            # print(len(seq))
            if len(seq) % 20 != 0:
                break
            optimizer.zero_grad()
            y_pred = model(seq).squeeze()  # 压缩维度：得到输出，并将维度为1的去除
            single_loss = loss_function(y_pred, seq)
            # 若想要获得类别，二分类问题使用四舍五入的方法即可：print(torch.round(y_pred))
            single_loss.backward()
            optimizer.step()
            # print("Train Step:", i, " loss: ", single_loss)
        # 每20次，输出一次前20个的结果，对比一下效果
        if i % 20 == 0:
            test_data = data[:20]
            y_pred = model(test_data).squeeze()  # 压缩维度：得到输出，并将维度为1的去除
            # print("TEST: ", test_data)
            # print("PRED: ", y_pred)
            print("LOSS: ", loss_function(y_pred, test_data))

    model.eval()
    vae_test = vae.dataprocess(vae_test[:10000])
    torch_vae = Data.TensorDataset(vae_test, vae_test)
    test_loader = Data.DataLoader(dataset=torch_vae, batch_size=20, shuffle=False)
    loss_f = torch.nn.MSELoss()
    prob = []
    for seq, ordata in test_loader:
        out = model(seq)
        for i, j in zip(out, ordata):
            prob.append(loss_f(i, j).detach().item())
    auc = metrics.roc_auc_score(label[:10000], prob)
    print('auc', auc)


if __name__ == '__main__':
    # VAE_test()
    # lstm_test()
    lstm_vae()
    # discriminator = exp1.classifier('LSTM', 1, 'no', True, 2)
