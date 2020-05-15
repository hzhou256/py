import numpy as np
import membership, Fuzzy_SVM
from sklearn import metrics
from imblearn.metrics import specificity_score
from sklearn.model_selection import cross_validate, StratifiedKFold


def Resplit(X, y): 
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

C_list = [0.125, 4.0, 8.0, 2.0, 1024.0, 16.0, 32.0]
g_list = [0.0078125, 0.001953125, 6.103515625e-05, 0.0625, 3.05176e-05, 0.0625, 1.0]
nu = 0.1
dataset = ['australian', 'breastw', 'diabetes', 'german', 'heart', 'ionosphere', 'sonar']
for i in range(6, 7):
    name = dataset[i]
    print(name)
    f1 = np.loadtxt('E:/Study/Bioinformatics/UCI/' + name + '/data.csv', delimiter = ',')
    X = f1[:, 0:-1]
    y = f1[:, -1]

    X_train, y_train = Resplit(X, y)

    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)

    C = C_list[i]
    gamma = g_list[i]

    clf = Fuzzy_SVM.FSVM_Classifier(C = C, gamma = gamma, membership = 'SVDD', nu = nu, proj = 'normal')
    clf.fit(X_train, y_train)

