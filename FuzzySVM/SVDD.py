import numpy as np
import collections
from scipy.spatial.distance import cdist
import numba
from cvxopt import matrix, solvers


@numba.jit(nopython = True, fastmath = True) 
def gaussian(vec1, vec2, g):
    k = np.exp(-g*np.square((np.linalg.norm(vec1 - vec2))))
    return k

def G(X, Y, g):
    K = cdist(X, Y, gaussian, g = g)
    return K

def split(X, y): 
    '''
    Divide dataset X into two subsets according to their labels y
    '''
    cnt = dict(collections.Counter(y))
    label = np.zeros(2)
    num = np.zeros(2)
    i = 0
    for key, val in cnt.items():
        label[i] = key
        num[i] = val
        i = i + 1
    n_pos = int(num[0])
    n_neg = int(num[1])
    n = np.shape(X)[1]
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

def QQP_solver(Gram, C):
    l = np.shape(Gram)[0]
    P = 2 * matrix(Gram)
    diag = np.zeros((l, 1))
    for i in range(l):
        diag[i] = Gram[i][i]
    q = matrix(diag)
    i = np.identity(l)
    G = matrix(np.row_stack((-i, i)))
    Zeros = np.zeros((l, 1))
    Cs = np.zeros((l, 1))
    for k in range(l):
        Cs[k] = C
    h = matrix(np.row_stack((Zeros, Cs)))
    Ones = np.zeros((l, 1))
    for j in range(l):
        Ones[j] = 1
    A = matrix(Ones.T)
    b = matrix(1.0)
    sol = solvers.qp(P, q, G, h, A, b)
    #print('x\n', sol['x'])
    return sol['x']

def get_distance_2(index, Gram, alpha): 
    '''
    Calculate distance between two points in feature space
    '''
    temp_1 = Gram[index][index]
    temp_2 = alpha[index] * np.sum(Gram[index])
    temp_3 = np.dot(alpha.T, Gram).dot(alpha)
    print(temp_1, temp_2, temp_3)
    d_square = temp_1 - 2 * temp_2 + temp_3
    return d_square

dataset_name = ['Antifp_Main', 'Antifp_DS1', 'Antifp_DS2']
for ds in range(1):
    name_ds = dataset_name[ds]
    print(name_ds)
    methods_name = ['AAC']
    for it in range(1):
        name = methods_name[it]

        f1 = np.loadtxt('E:/Study/Bioinformatics/AFP/kernel_matrix/' + name_ds + '/KM_train_tanimoto/KM_tanimoto_' + name + '_train.csv', delimiter = ',')
        gram = f1
        alpha = QQP_solver(gram, 5)
        #dist = np.reshape([get_distance_2(i, gram, alpha) for i in range(np.shape(gram)[0])], (np.shape(gram)[0], 1))
        #print(dist)
        d = get_distance_2(0, gram, alpha)
        print(d)
