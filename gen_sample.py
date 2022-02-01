from generate_sample.generator import Generator
from tqdm import tqdm
import torch
# from memory_profiler import profile
alphabet = [' ', 'g', 'o', 'l', 'e', 'y', 'u', 't', 'b', 'm', 'a', 'i', 'd', 'q', 's', 'h', 'f', 'c', 'k', '3', '6',
            '0', 'j', 'z', 'n', 'w', 'p', 'r', 'x', 'v', '1', '8', '7', '2', '9', '-', '5', '4', '_']

int_to_char = dict((i, c) for i, c in enumerate(alphabet))
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
g_emb_dim = 32
g_hidden_dim = 64
g_sequence_len = 100
VOCAB_SIZE = 38
batch_size = 64
cuda = False

@profile
def generate_sample(model_name, generated_num, exp_num, dga):
    generator = Generator(VOCAB_SIZE, g_emb_dim, g_hidden_dim, cuda)
    if exp_num is 1:
        # generator.load_state_dict(torch.load('./model/exp1/' + str(model_name) + '-collision.trc'))
        generator.load_state_dict(torch.load('./model/exp1/' + str(model_name) + '.trc'))
    elif exp_num is 2:
        generator.load_state_dict(torch.load('./model/exp2/' + str(model_name) + '-' + str(dga) + '.trc'))
    samples = []
    for _ in range(int(generated_num / batch_size)):
        sample = generator.sample(batch_size, g_sequence_len).cpu().data.numpy().tolist()
        samples.extend(sample)
    new_str = []
    for s in samples:
        new_int = []
        for b in s:
            if (int_to_char[b] == ' '):
                continue
            new_int.append(int_to_char[b])
        str1 = ''.join(new_int)
        if len(str1) > 2:
            new_str.append(str1)
    print(new_str)
    new_str = list(set(new_str))

    print(len(new_str))
    # f_train = ''
    # f_test = ''
    # if exp_num is 1:
    #     f_train = open('C:/Users/sxy/Desktop/experiment/dataset/our/2/exp1-' + str(model_name) + '-noc-train.txt', 'a')
    #     f_test = open('C:/Users/sxy/Desktop/experiment/dataset/our/2/exp1-' + str(model_name) + '-noc-test.txt', 'a')
    # elif exp_num is 2:
    #     f = open('C:/Users/sxy/Desktop/experiment/dataset/our/exp2-' +
    #              str(model_name) + '-' + str(dga) + '-test.txt', 'a')
    # for i in range(0, 100000):
    #     f_train.write(new_str[i] + '\n')
    # for i in range(len(new_str)-10000, len(new_str)):
    #     f_test.write(new_str[i] + '\n')
    # f_train.close()
    # f_test.close()


for i in ['LSTM']:
    generate_sample(i, 1000, 1, 'no')
# dga = ['matsnu', 'gozi', 'suppobox', 'pykspa', 'khaos', 'charbot']
# for i in dga:
#     generate_sample('RF', 1100, 2, i)
