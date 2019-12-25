import numpy as np


def get_matrix(file):
    m = np.shape(file)[0]
    n = np.shape(file)[1]
    d = np.zeros((m, n-1))
    for index in range(m):
        d[index] = file[index][1:]
    return d

def euclidean(vec1, vec2):
    d = np.linalg.norm(vec1 - vec2)
    return d

f1 = np.loadtxt('D:/下载/train_AAC_positive.csv', delimiter = ',', skiprows = 1)
f2 = np.loadtxt('D:/下载/train_AAC_negative.csv', delimiter = ',', skiprows = 1)

AAC_pos = get_matrix(f1)
AAC_neg = get_matrix(f2)

m = np.shape(AAC_neg)[0]
n = np.shape(AAC_neg)[1]

result = dict()

for i in range(m):
    print('负样本: ', i)
    min_dist = np.inf
    pos_index = -1
    for j in range(m):
        neg = AAC_neg[i]
        pos = AAC_pos[j]
        dist = euclidean(neg, pos)
        if dist < min_dist:
            min_dist = dist
            pos_index = j
    result.update({i:pos_index})

file_pos = open('D:/下载/train_positive.fasta', 'r')
file_neg = open('D:/下载/train_negative.fasta', 'r')

line_pos = file_pos.readlines()
line_neg = file_neg.readlines()


sequence = list()
for k, v in result.items():
    index_neg = (k+1)*2-1
    index_pos = (v+1)*2-1
    temp = line_neg[index_neg].strip('\n') + '  ' + line_pos[index_pos]
    sequence.append(temp)
#print(sequence)

new_file = open('D:/下载/similar.fasta', 'w')
for line in sequence:
    new_file.writelines(line)
new_file.close()
file_neg.close()
file_pos.close()