import numpy as np


f1 = open('D:/Study/生物信息学/Kernel PCA/群体反应信号肽/datasets/datasets.fasta')
f2 = open('D:/Study/生物信息学/Kernel PCA/群体反应信号肽/datasets/ACC.txt', 'w')

AA = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def get_freq(s, k):
    l = np.zeros(20)
    d = len(s)
    for x in s:
        for i in range(20):
            if x == AA[i]:
                l[i] = l[i] + 1
    if k == 1:
        for i in range(20):
            l[i] = l[i] / d
    np.savetxt(f2, l, fmt='%.8e ', newline='', header='[', footer=']')

for line in f1:
    if not line.startswith('>'):
        get_freq(line, 1)
        f2.write('\n')


f1.close()
f2.close()
