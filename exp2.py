import exp1

if __name__ == '__main__':
    # test_classifier()
    # name = ['LSTM', 'CNN']
    name = ['bilstm']
    cuda = False
    # dga = ['khaos_original', 'kraken', 'gozi', 'suppobox', 'maskDGA', 'our']
    dga_test = ['maskDGA']
    for model in name:
        exp1.classification(model, 1, ' ', dga_test, cuda)
    # classification('Graph', 1, ['no'])