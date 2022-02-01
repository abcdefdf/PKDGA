import os
import sys
import time
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import khaos.language_helpers as language_helpers
import khaos.tflib as lib
from sklearn.preprocessing import OneHotEncoder
import khaos.N_gram as N_gram
import lstm.lstm_float as lstm
import sklearn.metrics as metrics
import exp1

sys.path.append(os.getcwd())
torch.manual_seed(1)
# use_cuda = torch.cuda.is_available()
use_cuda = False
if use_cuda:
    gpu = 0

# Download Google Billion Word at http://www.statmt.org/lm-benchmark/ and
# fill in the path to the extracted files here!
DATA_DIR = 'benign.txt'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_language.py!')

BATCH_SIZE = 64  # Batch size
ITERS = 55  # How many iterations to train for
SEQ_LEN = 10  # Sequence length in characters
DIM = 64  # Model dimensionality. This is fairly slow and overfits, even on
          # Billion Word. Consider decreasing for smaller datasets.
CRITIC_ITERS = 156 # How many critic iterations per generator iteration. We
                  # use 10 for the results in the paper, but 5 should work fine
                  # as well.
LAMBDA = 10  # Gradient penalty lambda hyperparameter.
MAX_N_EXAMPLES = 10000  # 10000000 # Max number of data examples to load. If data loading
                          # is too slow or takes too much RAM, you can decrease
                          # this (at the expense of having less training data).


lib.print_model_settings(locals().copy())
# line 每行数据 charmap 字符对应表, inv_charmap 字符表
lines, charmap, inv_charmap = language_helpers.load_dataset(
    max_length=SEQ_LEN,
    max_n_examples=MAX_N_EXAMPLES,
    data_dir=DATA_DIR
)
table = np.arange(5000).reshape(-1, 1)
one_hot = OneHotEncoder()
one_hot.fit(table)
# ==================Definition Start======================


def make_noise(shape, volatile=False):
    tensor = torch.randn(shape).cuda(gpu) if use_cuda else torch.randn(shape)
    return autograd.Variable(tensor, volatile)


class ResBlock(nn.Module):

    def __init__(self):
        super(ResBlock, self).__init__()

        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(DIM, DIM, 3, padding=1),  # nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Conv1d(DIM, DIM, 3, padding=1),  # nn.Linear(DIM, DIM),
        )

    def forward(self, input):
        output = self.res_block(input)
        return input + (0.3*output)


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(5000, DIM*SEQ_LEN)

        self.block = nn.Sequential(
            ResBlock(),
            ResBlock(),
            ResBlock(),
        )
        self.conv1 = nn.Conv1d(DIM, 5000, 1)
        self.softmax = nn.Softmax()

    def forward(self, noise):
        output = self.fc1(noise)
        output = output.view(-1, DIM, SEQ_LEN)  # (BATCH_SIZE, DIM, SEQ_LEN)
        output = self.block(output)
        output = self.conv1(output)
        output = output.transpose(1, 2)
        shape = output.size()
        output = output.contiguous()
        output = output.view(BATCH_SIZE*SEQ_LEN, -1)
        output = self.softmax(output)
        return output.view(shape)  # (BATCH_SIZE, SEQ_LEN, len(charmap))


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1d = nn.Conv1d(5000, DIM, 1)
        self.block = nn.Sequential(
            ResBlock(),
            ResBlock(),
            ResBlock(),
        )
        self.linear = nn.Linear(SEQ_LEN*DIM, 1)

    def forward(self, input):
        output = input.transpose(1, 2)  # (BATCH_SIZE, len(charmap), SEQ_LEN)
        output = self.conv1d(output)
        output = self.block(output)
        output = output.view(-1, SEQ_LEN*DIM)
        output = self.linear(output)
        return output


class LSTM(nn.Module):
    def __init__(self, num_class, vocab_size, emb_dim, hidden_dim, use_cuda, num_layers):
        super(LSTM, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_cuda = use_cuda

        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        # self.lstm = nn.LSTM(vocab_size, hidden_dim, num_layers=num_layers, batch_first=True)
        self.lin = nn.Linear(hidden_dim, num_class)

        self.init_params()

    def init_hidden(self, batch_size):
        h = autograd.Variable(torch.zeros((self.num_layers, batch_size, self.hidden_dim)))
        c = autograd.Variable(torch.zeros((self.num_layers, batch_size, self.hidden_dim)))
        if self.use_cuda:
            h, c = h.cuda(), c.cuda()
        return h, c

    # def forward(self, x, data_len):
    def forward(self, x):
        self.lstm.flatten_parameters()
        emb = self.emb(x)
        h0, c0 = self.init_hidden(x.size(0))
        output, (h, c) = self.lstm(emb, (h0, c0))
        # output, (h, c) = self.lstm(x, (h0, c0))
        output = output[:, -1, :]
        output = self.lin(output.contiguous().view(-1, self.hidden_dim))
        pred = F.softmax(output, dim=1)
        return pred

    def init_params(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)


# Dataset iterator
def inf_train_gen():
    while True:
        np.random.shuffle(lines)
        for i in range(0, len(lines)-BATCH_SIZE+1, BATCH_SIZE):
            yield np.array(
                [[charmap[c] for c in l] for l in lines[i:i+BATCH_SIZE]]
            )


def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1, 1)
    # alpha = torch.rand(BATCH_SIZE, 10)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda(gpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    # interpolate = interpolates.long()
    if use_cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    # interpolate = autograd.Variable(interpolate, requires_grad=True)
    disc_interpolates = netD(interpolates)
    # disc_interpolates = netD(interpolate)
    # TODO: Make ConvBackward diffentiable
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def generate_samples(netG, Sample_num):
    noise = torch.randn(Sample_num, 5000)
    if use_cuda:
        noise = noise.cuda(gpu)
    # with torch.no_grad():
        # noisev = autograd.Variable(noise)
    noisev = autograd.Variable(noise, volatile=True)  # totally freeze netG
    samples = netG(noisev)
    samples = samples.view(-1, SEQ_LEN, 5000)
    samples = samples.cpu().data.numpy()
    samples = np.argmax(samples, axis=2)
    N_gram.int_to_char(samples)

    # return decoded_samples


def orignal_to_lstm(input):
    int_emb = []
    for i in input:
        str1 = []
        for l in i:
            for j in range(len(l)):
                if l[j] is 1:
                    str1.append(j)
        int_emb.append(str1)
    return int_emb

# ==================Definition End======================


netG = Generator()
# netD = Discriminator()
# netD = LSTM(1, 5000, 32, 32, use_cuda, 1)
netD = lstm.Discriminator()
print(netG)
print(netD)
data_one_hot_all, MAX_LENA, dictionary = N_gram.Ont_hot()
print(MAX_LENA)
if use_cuda:
    netD = netD.cuda(gpu)
    netG = netG.cuda(gpu)

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

# one = torch.FloatTensor([1])
one = torch.tensor(1, dtype=torch.float)
mone = one * -1
if use_cuda:
    one = one.cuda(gpu)
    mone = mone.cuda(gpu)

# data = inf_train_gen()
# During training we monitor JS divergence between the true & generated ngram
# distributions for n=1,2,3,4. To get an idea of the optimal values, we
# evaluate these statistics on a held-out set first.
true_char_ngram_lms = [language_helpers.NgramLanguageModel(i+1, lines[10*BATCH_SIZE:], tokenize=False) for i in range(4)]
validation_char_ngram_lms = [language_helpers.NgramLanguageModel(i+1, lines[:10*BATCH_SIZE], tokenize=False) for i in range(4)]
for i in range(4):
    print("validation set JSD for n={}: {}".format(i+1, true_char_ngram_lms[i].js_with(validation_char_ngram_lms[i])))
true_char_ngram_lms = [language_helpers.NgramLanguageModel(i+1, lines, tokenize=False) for i in range(4)]
AUC_c = []
for iteration in range(ITERS):
    print("iteration is %d" % iteration)
    start_time = time.time()
    ############################
    # (1) Update D network
    ###########################
    for p in netD.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update
    for iter_d in range(CRITIC_ITERS):
        # _data = next(data)
        # data_one_hot = one_hot.transform(_data.reshape(-1, 1)).toarray().reshape(BATCH_SIZE, -1, len(charmap))
        data_one_hot = data_one_hot_all[iter_d*BATCH_SIZE: (iter_d+1)*BATCH_SIZE]
        # int_emb = orignal_to_lstm(data_one_hot)
        # print data_one_hot.shape
        # 释放无关内存
        real_data = torch.Tensor(data_one_hot)
        # real_data = torch.LongTensor(int_emb)
        if use_cuda:
            real_data = real_data.cuda(gpu)
        real_data_v = autograd.Variable(real_data)

        netD.zero_grad()
        # train with real
        D_real_i = netD(real_data_v)
        D_real = D_real_i.mean()
        # print D_real
        # TODO: Waiting for the bug fix from pytorch
        D_real.backward(mone)

        # train with fake
        # noise = torch.randn(BATCH_SIZE, 5000)
        # if use_cuda:
        #     noise = noise.cuda(gpu)
        # noisev = autograd.Variable(noise)
        # fake = netG(noisev)
        # fake = fake.view(-1, SEQ_LEN, 5000)
        # fake = fake.data.numpy()
        # fake = np.argmax(fake, axis=2)
        # fake = torch.tensor(fake)
        # fake = autograd.Variable(fake)
        # inputv = fake
        # D_fake = netD(inputv)
        # D_fake = D_fake.mean()
        # # TODO: Waiting for the bug fix from pytorch
        # D_fake.backward(one)
        noise = torch.randn(BATCH_SIZE, 5000)
        if use_cuda:
            noise = noise.cuda(gpu)
        # with torch.no_grad():
            # noisev = autograd.Variable(noise) # totally freeze netG
        noisev = autograd.Variable(noise, volatile=True)  # totally freeze netG
        fake = autograd.Variable(netG(noisev).data)
        inputv = fake
        D_fake_i = netD(inputv)
        D_fake = D_fake_i.mean()
        # TODO: Waiting for the bug fix from pytorch
        D_fake.backward(one)
        # label = [0] * len(real_data_v.data) + [1] * len(fake.data)
        # pred = torch.cat([D_real_i, D_fake_i], dim=0)
        # auc = metrics.roc_auc_score(label, pred.detach().numpy())
        # AUC_c.append(auc)
        # auc = exp1.get_auc(netD)
        # print(auc)
        # train with gradient penalty
        gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data)
        gradient_penalty.backward()

        D_cost = D_fake - D_real + gradient_penalty
        Wasserstein_D = D_real - D_fake
        optimizerD.step()
    # auc = exp1.get_auc(netD)
    # print(auc)
    ############################
    # (2) Update G network
    ###########################
    print("update G network")
    for p in netD.parameters():
        p.requires_grad = False  # to avoid computation
    netG.zero_grad()

    noise = torch.randn(BATCH_SIZE, 5000)
    if use_cuda:
        noise = noise.cuda(gpu)
    noisev = autograd.Variable(noise)
    fake = netG(noisev)
    G = netD(fake)
    G = G.mean()
    G.backward(mone)
    G_cost = -G
    optimizerG.step()
    print("G end")
    # print(AUC_c)
    # Write logs and save samples
    '''lib.plot.plot('tmp/lang/time', time.time() - start_time)
    lib.plot.plot('tmp/lang/train disc cost', D_cost.cpu().data.numpy())
    lib.plot.plot('tmp/lang/train gen cost', G_cost.cpu().data.numpy())
    lib.plot.plot('tmp/lang/wasserstein distance', Wasserstein_D.cpu().data.numpy())

    if iteration % 100 == 99:
        samples = []
        for i in range(10):
            samples.extend(generate_samples(netG))

        for i in range(4):
            lm = language_helpers.NgramLanguageModel(i+1, samples, tokenize=False)
            lib.plot.plot('tmp/lang/js{}'.format(i+1), lm.js_with(true_char_ngram_lms[i]))

        with open('tmp/lang/samples_{}.txt'.format(iteration), 'w') as f:
            for s in samples:
                s = "".join(s)
                f.write(s + "\n")

    if iteration % 100 == 99:
        lib.plot.flush()'''
generate_samples(netG, 11000)
torch.save(netG.state_dict(), './khaos.trc')
