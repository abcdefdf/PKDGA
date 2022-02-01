# from khaos.gan_language import Generator
import torch
import numpy as np
import torch.autograd as autograd
import khaos.N_gram as N_gram
from memory_profiler import profile
import torch.nn as nn

use_cuda = False
if use_cuda:
    gpu = 0
Sample_num = 10000
SEQ_LEN = 10
BATCH_SIZE = 64  # Batch size
ITERS = 55  # How many iterations to train for
# SEQ_LEN = 10  # Sequence length in characters
DIM = 64  # Model dimensionality. This is fairly slow and overfits, even on
          # Billion Word. Consider decreasing for smaller datasets.
CRITIC_ITERS = 156 # How many critic iterations per generator iteration. We
                  # use 10 for the results in the paper, but 5 should work fine
                  # as well.
LAMBDA = 10  # Gradient penalty lambda hyperparameter.
MAX_N_EXAMPLES = 10000  # 10000000 # Max number of data examples to load. If data loading
                          # is too slow or takes too much RAM, you can decrease
                          # this (at the expense of having less training data).
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


@profile
def gen():
    netG = Generator()
    netG.load_state_dict(torch.load('./khaos.trc'))
    noise = torch.randn(Sample_num, 5000)
    # noise = noise.cuda(gpu)
    with torch.no_grad():
        noisev = autograd.Variable(noise, volatile=True)  # totally freeze netG
    print(noisev)
    samples = netG(noisev)
    samples = samples.view(-1, SEQ_LEN, 5000)
    samples = samples.cpu().data.numpy()
    samples = np.argmax(samples, axis=2)
    new_str = N_gram.int_to_char(samples)
    print(new_str)
    # f = open('./khaos.txt', 'a')
    # for i in new_str:
    #     f.write(i + '\n')
    # f.close()

gen()
