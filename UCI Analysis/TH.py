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

def QP_solver(Gram_self, Gram_cross, C, v):
    l_1 = np.shape(Gram_self)[0]
    l_2 = np.shape(Gram_cross)[1]
    P = 2 * matrix(Gram_self)
    diag = np.zeros((l_1, 1))
    for i in range(l_1):
        diag[i] = Gram_self[i][i]
    Sum_cross = np.reshape(np.sum(Gram_cross, axis = 1), (l_1, 1))
    q = matrix(-(2*v/l_2) * Sum_cross - (1-v) * diag)
    i = np.identity(l_1)
    G = matrix(np.row_stack((-i, i)))
    Zeros = np.zeros((l_1, 1))
    Cs = np.zeros((l_1, 1))
    for k in range(l_1):
        Cs[k] = C/v
    h = matrix(np.row_stack((Zeros, Cs)))
    Ones = np.zeros((l_1, 1))
    for j in range(l_1):
        Ones[j] = 1.0
    A = matrix(Ones.T)
    b = matrix(1.0)
    sol = solvers.qp(P, q, G, h, A, b)
    #print('x\n', sol['x'])
    return sol['x']

def get_index_set(alpha, C):
    index_set = []
    l = np.shape(alpha)[0]
    for i in range(l):
        if alpha[i][0] > 0 and alpha[i][0] < (C/l):
            index_set.append(i)
    return index_set

def get_distance_2(index, Gram_self, Gram_cross, Gram_other, alpha, v): 
    l_1 = np.shape(Gram_self)[0]
    l_2 = np.shape(Gram_cross)[1]
    temp_1 = Gram_self[index][index]
    temp_2 = np.dot(alpha.T, np.reshape(Gram_self[index], (l_1, 1)))
    temp_3 = np.sum(Gram_cross[index])
    sum_temp = np.reshape(np.sum(Gram_cross, axis = 1), (l_1, 1))
    temp_4 = np.dot(alpha.T, Gram_self).dot(alpha) - (2*v/l_2)*np.dot(alpha.T, sum_temp) + (v/l_2)**2 * np.asmatrix(Gram_other).sum()
    d_square = temp_1 - 2/(1-v) * temp_2 + 2*v/l_2 * temp_3 + ((1/(1-v))**2) * temp_4
    return d_square

def get_radius_2(index_set, Gram_self, Gram_cross, Gram_other, alpha, v):
    L = len(index_set)
    Sum = 0
    for index in index_set:
        Sum += get_distance_2(index, Gram_self, Gram_cross, Gram_other, alpha, v)
    R_2 = Sum / L
    return R_2

def TH_membership(X, y, g, C_1, C_2, v_1, v_2):
    X_pos, X_neg = split(X, y)
    G_pos = G(X_pos, X_pos, g)
    G_neg = G(X_neg, X_neg, g)
    G_pos_neg = G(X_pos, X_neg, g)
    G_neg_pos = G(X_neg, X_pos, g)
    n_pos = np.shape(X_pos)[0]
    n_neg = np.shape(X_neg)[0]

    try:
        alpha_pos = np.reshape(QP_solver(G_pos, G_pos_neg, C_1, v_1), (n_pos, 1))
        alpha_neg = np.reshape(QP_solver(G_neg, G_neg_pos, C_2, v_2), (n_neg, 1))
    except ValueError:
        return []

    index_set_pos = get_index_set(alpha_pos, C_1)
    index_set_neg = get_index_set(alpha_neg, C_2)

    R_2_pos = get_radius_2(index_set_pos, G_pos, G_pos_neg, G_neg, alpha_pos, v_1)
    R_2_neg = get_radius_2(index_set_neg, G_neg, G_neg_pos, G_pos, alpha_neg, v_2)

    D_2_pos = np.reshape([get_distance_2(i, G_pos, G_pos_neg, G_neg, alpha_pos, v_1) for i in range(n_pos)], (n_pos, 1))
    D_2_neg = np.reshape([get_distance_2(j, G_neg, G_neg_pos, G_pos, alpha_neg, v_2) for j in range(n_neg)], (n_neg, 1))

    s_pos = np.reshape([np.sqrt(D_2_pos[i] / R_2_pos) for i in range(n_pos)], (n_pos, 1)) # 有问题
    s_neg = np.reshape([np.sqrt(D_2_neg[j] / R_2_neg) for j in range(n_neg)], (n_neg, 1))
    s = np.row_stack((s_neg, s_pos))
    return s

