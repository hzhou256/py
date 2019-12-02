#polar = ['N','Q','S','D','E','C','T','K','R','H','Y','W']
#positive = ['K','H','R']
#negative = ['D','E']
#charged = ['K','H','R','D','E']
#hydrophobic = ['A','G','C','T','I','V','L','K','H','F','Y','W','M']
#aliphatic = ['I','V','L']
#aromatic = ['F','Y','W','H']
#small = ['P','N','D','T','C','A','G','S','V']
#tiny = ['A','S','G','C']
#proline = ['P']

import numpy as np
import csv


total = [['N','Q','S','D','E','C','T','K','R','H','Y','W'],
    ['K','H','R'],
    ['D','E'],
    ['K','H','R','D','E'],
    ['A','G','C','T','I','V','L','K','H','F','Y','W','M'],
    ['I','V','L'],
    ['F','Y','W','H'],
    ['P','N','D','T','C','A','G','S','V'],
    ['A','S','G','C'],
    ['P']]

AA = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

M = np.zeros((20,10))
for i in range(20):
    for j in range(10):
        if AA[i] in total[j]:
            M[i][j] = 1
        else:
            M[i][j] = 0

N = np.zeros((400,50))
def get_vec(seq, N, k):
    index = 0
    for i in range(5):
        cnt = 0
        for j in range(20):
            if seq[i] == AA[j]:
                break
            cnt = cnt + 1
        for x in range(10):
            N[k][index] = M[cnt][x]
            index = index + 1
        
f1 = open('D:/Study/生物信息学/Kernel_PCA/群体反应信号肽/datasets/21-bit/N端(左).fasta')
line = f1.readlines()
index = 0
for line_list in line:
    if not line_list.startswith('>'):
        line_new = line_list.strip('/n')
        get_vec(line_new, N, index)
        index = index + 1

with open('D:/Study/生物信息学/Kernel_PCA/群体反应信号肽/datasets/OVP/N端(左).csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in N:
        writer.writerow(row)

f1.close()
csvfile.close()
