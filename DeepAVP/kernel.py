import numpy as np
import csv
from scipy.spatial.distance import cdist

def tanimoto_base(p_vec, q_vec):
    pq = np.dot(p_vec, q_vec)
    p_square = np.square(np.linalg.norm(p_vec))
    q_square = np.square(np.linalg.norm(q_vec))
    d = pq / (p_square + q_square - pq)
    return d

def tanimoto(X, Y):
    K = cdist(X, Y, tanimoto_base)
    return K

def split(X, y): 
    '''
    y = {-1, 1}
    '''
    n_pos, n_neg = 0, 0
    for y_i in y:
        if y_i == 1:
            n_pos += 1
        elif y_i == -1:
            n_neg += 1
    n = np.shape(X)[1]
    X_pos = np.zeros((n_pos, n)) 
    X_neg = np.zeros((n_neg, n)) 
    j, k = 0, 0
    for i in range(n_pos + n_neg):
        if y[i] == -1:
            X_neg[j] = X[i]
            j = j + 1
        else:
            X_pos[k] = X[i]
            k = k + 1
    y = np.zeros(n_neg + n_pos)
    for i in range(n_neg):
        y[i] = -1
    for i in range(n_neg, n_neg + n_pos):
        y[i] = 1
    X = np.row_stack((X_neg, X_pos))
    return X, y

f1 = np.loadtxt('E:/Study/Bioinformatics/DeepAVP/DPC/train_DPC.csv', delimiter = ',')
X_train = f1[:, 0:-1]
y_train = f1[:, -1]
X_train, y_train = split(X_train, y_train)

f2 = np.loadtxt('E:/Study/Bioinformatics/DeepAVP/DPC/test_DPC.csv', delimiter = ',')
X_test = f2[:, 0:-1]
y_test = f2[:, -1]

Gram_train = tanimoto(X_train, X_train)
Gram_test = tanimoto(X_test, X_train)

with open('E:/Study/Bioinformatics/DeepAVP/DPC/gram_train_DPC.csv', 'w', newline = '') as csvfile:
    writer = csv.writer(csvfile)
    for row in Gram_train:
        writer.writerow(row)
    csvfile.close()

with open('E:/Study/Bioinformatics/DeepAVP/DPC/gram_test_DPC.csv', 'w', newline = '') as csvfile:
    writer = csv.writer(csvfile)
    for row in Gram_test:
        writer.writerow(row)
    csvfile.close()