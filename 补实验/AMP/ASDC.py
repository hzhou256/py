import numpy as np
import csv


AA = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

f1 = open('D:\\Study\\Bioinformatics\\补实验\\AMP\\fasta\\train.fasta')
line = f1.readlines()

n_protein = int(len(line) / 2)
print(n_protein)

ASDC = np.zeros((n_protein, 400))

def get_ASDC(seq):
    Mat = np.zeros(400)
    sum = 0
    index = 0
    for i in range(20):
        for j in range(20):
            X = AA[i]
            Y = AA[j]
            cnt_pair = 0
            m = 0
            l = len(seq)
            while m < l:
                n = m + 1
                if seq[m] == X:
                    while n < l and n != m:
                        if seq[n] == Y:
                            cnt_pair = cnt_pair + 1
                        n = n + 1
                m = m + 1
            sum = sum + cnt_pair
            Mat[index] = cnt_pair
            index = index + 1
    for p in range(400):
        Mat[p] = Mat[p] / sum
    return Mat

x = 0
for line_list in line:
    if not line_list.startswith('>'):
        line_new = line_list.strip('/n')
        ASDC[x] = get_ASDC(line_new)
        x = x + 1
print(ASDC)

with open('D:\\Study\\Bioinformatics\\补实验\\AMP\\features\\train_ASDC.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in ASDC:
        writer.writerow(row)

f1.close()
csvfile.close()
