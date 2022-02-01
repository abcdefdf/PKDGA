alphabet = [' ', 'g', 'o', 'l', 'e', 'y', 'u', 't', 'b', 'm', 'a', 'i', 'd', 'q', 's', 'h', 'f', 'c', 'k', '3', '6',
            '0', 'j', 'z', 'n', 'w', 'p', 'r', 'x', 'v', '1', '8', '7', '2', '9', '-', '5', '4', '_']

int_to_char = dict((i, c) for i, c in enumerate(alphabet))
char_to_int = dict((c, i) for i, c in enumerate(alphabet))

g_sequence_len = 65
benign_txt = 'C:/Users/sxy/Desktop/experiment/dataset/train/benign.txt'
malicious_txt = 'C:/Users/sxy/Desktop/experiment/dataset/train/malicious.txt'
