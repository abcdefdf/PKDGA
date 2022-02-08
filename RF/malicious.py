import os
import random


def get_csv(name):
    domain_c = []
    with open(name) as f:
        for line in f:
            line = line.split(',')
            temp = line[0].replace('"', '')
            temp = temp.replace('"', '')
            temp = temp.split('.')
            domain = temp[0]
            domain_c.append(domain)
        domain_c = list(set(domain_c))
    f.close()

    return len(domain_c), domain_c


def static():
    path = './DGArchive/'
    filelist = os.listdir(path)
    dga_len = 0
    lens = []
    malicious = []
    for i in filelist:
        name = path + i
        print(name)
        len1, domain_a = get_csv(name)
        if len1 < 2000:
            dga_len += 1
            for j in domain_a:
                malicious.append(j)
        else:
            resultlist = random.sample(range(0, len1), 2000)
            for j in resultlist:
                malicious.append(domain_a[j])
    print(dga_len)
    return malicious


if __name__ == '__main__':
    maliciouss = static()
    f = open('./new_malicious.txt', 'w')
    for i in maliciouss:
        f.write(i + '\n')
