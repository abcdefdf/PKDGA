import sklearn.metrics as metrics
import time
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Bidirectional, Embedding, LSTM
tensorflow.compat.v1.disable_eager_execution()
from tensorflow.keras.layers import Activation, Conv1D, GlobalMaxPooling1D, MaxPooling1D
from sklearn.model_selection import train_test_split
import file_check

train_benign = 70
train_malicious = 70
# train_malicious = 101337
# train_benign = 100000
test_benign = 1000
test_malicious = 1000
maxlen = 65
MAX_SEQ = 65

# benign_txt = "./dataset/top-100000-domains"
# malicious_txt = "./100000_m.txt"
# benign_txt = "./dataset/okk.txt"
# malicious_txt = "./dataset/bad.txt"
alphabet = [' ', 'g', 'o', 'l', 'e', 'y', 'u', 't', 'b', 'm', 'a', 'i', 'd', 'q', 's', 'h', 'f', 'c', 'k', '3', '6',
            '0', 'j', 'z', 'n', 'w', 'p', 'r', 'x', 'v', '1', '8', '7', '2', '9', '-', '5', '4', '_']
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
char_to_int = dict((c, i) for i, c in enumerate(alphabet))


def bidirectional_lstm_model(max_features, maxlen):
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    return model


def build_model(max_features, maxlen):
    """Build LSTM model"""
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen))
    '''
    keras.layers.Embedding(input_dim, output_dim, input_length)
    input_dim：这是文本数据中词汇的取值可能数。例如，如果您的数据是整数编码为0-9之间的值，那么词汇的大小就是10个单词；
    output_dim：这是嵌入单词的向量空间的大小。它为每个单词定义了这个层的输出向量的大小。例如，它可能是32或100甚至更大，
                可以视为具体问题的超参数；
    input_length：这是输入序列的长度，就像您为Keras模型的任何输入层所定义的一样，也就是一次输入带有的词汇个数。
                  例如，如果您的所有输入文档都由1000个字组成，那么input_length就是1000。
    '''
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop')

    return model


def basic_cnn_model(max_features, maxlen):

    filters = 250
    kernel_size = 3
    hidden_dims = 250
    batch_size = 256

    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop')
    return model


def cnn_lstm_model(max_features, maxlen):

    kernel_size = 5
    filters = 64
    pool_size = 4

    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(LSTM(128))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop')
    return model


def get_data(domain, label):
    data = []
    for i in domain:
        temp = []
        for j in i.lower():
            temp.append(char_to_int[j])
        if len(temp) < MAX_SEQ:
            zeros = [0] * (MAX_SEQ - len(temp))
            temp = zeros + temp
        data.append(temp)
    data = np.array(data)
    label = np.array(label)
    return data, label


def train(x_train, y_train, dis_num):
    x_train, y_train = get_data(x_train, y_train)
    max_features = len(char_to_int) + 1
    model = bidirectional_lstm_model(max_features, maxlen)
    print("Train...")
    model.fit(x_train, np.array(y_train), batch_size=128, epochs=dis_num)
    return model


def predict(model, test):
    # test = test.to(torch.int64)
    test = test.cpu().numpy()
    t_probs = model.predict(test)
    print(t_probs)
    return t_probs
