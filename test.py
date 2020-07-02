import numpy as np
import collections

'''
def split_X_by_label(X, y):
    cnt = dict(collections.Counter(y))
    n_class = len(cnt)
    label = np.zeros(n_class)
    num = np.zeros(n_class)
    i = 0
    for key, val in cnt.items():
        label[i] = key
        num[i] = int(val)
        i = i + 1
    n_features = np.shape(X)[1]
    X_pos = np.zeros((n_pos, n)) 
    X_neg = np.zeros((n_neg, n)) 
    j, k = 0, 0
    for i in range(n_pos + n_neg):
        if y[i] == label[1]:
            X_neg[j] = X[i]
            j = j + 1
        else:
            X_pos[k] = X[i]
            k = k + 1
    return X_pos, X_neg
'''

f1 = np.loadtxt('E:/Study/Bioinformatics/UCI/australian/data.csv', delimiter = ',')
X = f1[:, 0:-1]
y = f1[:, -1]

cnt = dict(collections.Counter(y))
n_class = len(cnt)
print(n_class)