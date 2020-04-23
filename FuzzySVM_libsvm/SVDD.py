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
        Ones[j] = 1.0
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
    temp_2 = np.dot(alpha.T, Gram[index].T)
    temp_3 = np.dot(alpha.T, Gram).dot(alpha)
    d_square = temp_1 - 2 * temp_2 + temp_3
    return d_square

def SVDD_membership(X, y, K, g, C):
    X_pos, X_neg = split(X, y)
    G_pos = G(X_pos, X_pos, g)
    G_neg = G(X_neg, X_neg, g)
    n_pos = np.shape(X_pos)[0]
    n_neg = np.shape(X_neg)[0]

    alpha_pos = np.reshape(QQP_solver(G_pos, C), (n_pos, 1))
    alpha_neg = np.reshape(QQP_solver(G_neg, C), (n_neg, 1))
    D_2_pos = np.reshape([get_distance_2(i, G_pos, alpha_pos) for i in range(n_pos)], (n_pos, 1))
    D_2_neg = np.reshape([get_distance_2(i, G_neg, alpha_neg) for i in range(n_neg)], (n_neg, 1))
    D_pos = np.sqrt(D_2_pos)
    D_neg = np.sqrt(D_2_neg)

    d_pos_max = np.max(D_pos)
    d_pos_min = np.min(D_pos)
    d_neg_max = np.max(D_neg)
    d_neg_min = np.min(D_neg)

    s_pos = np.reshape([np.sqrt((1 - (D_pos[i] - d_pos_min)/(d_pos_max - d_pos_min))) for i in range(n_pos)], (n_pos, 1))
    s_neg = np.reshape([np.sqrt((1 - (D_neg[i] - d_neg_min)/(d_neg_max - d_neg_min))) for i in range(n_neg)], (n_neg, 1))
    s = np.row_stack((s_neg, s_pos))
    print(np.shape(s))
    return s

