import numpy as np
import collections
from scipy.spatial.distance import cdist
import numba
from cvxopt import matrix, solvers
from sklearn import svm, preprocessing



delta = 0.001

@numba.jit(nopython = True, fastmath = True) 
def gaussian(vec1, vec2, g):
    k = np.exp(-g*np.square((np.linalg.norm(vec1 - vec2))))
    return k

def G(X, Y, g):
    K = cdist(X, Y, gaussian, g = g)
    return K

'''
def split(X, y): 
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
'''

def split(X, y): 
    '''
    Divide dataset X into two subsets according to their labels y (1, -1)
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
        elif y[i] == 1:
            X_pos[k] = X[i]
            k = k + 1
    return X_pos, X_neg


# 1. Use Class Center to Reduce the Effects of Outliers
def class_center_membership(X, y, delta): 
    '''
    X: data, y: label, delta: parameter
    '''
    cnt = collections.Counter(y)
    n_pos = cnt[1] 
    n_neg = cnt[0]
    X_pos, X_neg = split(X, y)
    mean_pos = np.mean(X_pos, axis = 0)
    mean_neg = np.mean(X_neg, axis = 0)
    temp_pos = np.zeros(n_pos)
    temp_neg = np.zeros(n_neg)
    for i in range(n_pos):
        temp_pos[i] = np.linalg.norm(mean_pos - X_pos[i])
    for i in range(n_neg):
        temp_neg[i] = np.linalg.norm(mean_neg - X_neg[i])
    r_pos = np.max(temp_pos)
    r_neg = np.max(temp_neg)
    s = np.zeros(len(y))
    for i in range(len(y)):
        if y[i] == 1:
            s[i] = 1 - np.linalg.norm(mean_pos - X[i])/(r_pos + delta)
        elif y[i] == 0:
            s[i] = 1 - np.linalg.norm(mean_neg - X[i])/(r_neg + delta)
    return s

# 2. Fuzzy membership function for nonlinear SVM

def get_radius_square(index, X, Gram, Sum, K, **kwargs): 
    '''
    Calculate distance between two points in feature space
    '''
    n = np.shape(X)[0]
    k_temp = Gram[index][index]
    temp_1 = np.sum(Gram[index])
    r_square = k_temp - 2/n * temp_1 + 1/(n*n) * Sum
    return r_square

def FSVM_2_membership(X, y, delta, K, **kwargs): 
    '''
    X: data, y: label, delta: parameter, K: kernel function
    '''
    X_pos, X_neg = split(X, y)
    G_pos = G(X_pos, X_pos, **kwargs)
    G_neg = G(X_neg, X_neg, **kwargs)
    sum_pos = np.asmatrix(G_pos).sum()
    sum_neg = np.asmatrix(G_neg).sum()
    r_square_pos = np.max([(get_radius_square(i, X_pos, G_pos, sum_pos, K, **kwargs)) for i in range(np.shape(X_pos)[0])])
    r_square_neg = np.max([(get_radius_square(i, X_neg, G_neg, sum_neg, K, **kwargs)) for i in range(np.shape(X_neg)[0])])
    n_pos = np.shape(X_pos)[0]
    n_neg = np.shape(X_neg)[0]
    s_pos = np.zeros(n_pos)
    s_neg = np.zeros(n_neg)
    for i in range(n_pos):
        s_pos[i] = 1 - np.sqrt(get_radius_square(i, X_pos, G_pos, sum_pos, K, **kwargs)/(r_square_pos + delta))
    for j in range(n_neg):
        s_neg[j] = 1 - np.sqrt(get_radius_square(j, X_neg, G_neg, sum_neg, K, **kwargs)/(r_square_neg + delta))
    s = np.hstack((s_neg, s_pos))
    return s

# 3. Fuzzy SVM for Noisy Data
def get_kth_evec(k, sorted_indices, evecs): 
    '''
    Calculate k_th largest eigenvector
    '''
    kth_evec = evecs[:,sorted_indices[-k]]
    return kth_evec

def get_k(threshold, sorted_indices, evals):
    evals_sort = np.sort(evals)
    evals_sort = evals_sort[::-1]
    Sum_evals = np.sum(evals_sort)
    Sum = 0
    for k in range(len(evals_sort)):
        Sum += evals_sort[k]
        if (Sum / Sum_evals) >= threshold:
            break
    return k+1

def get_beta(index, j, Gram, sorted_indices, evecs): 
    '''
    Calculate beta_j
    '''
    alpha = get_kth_evec(j, sorted_indices, evecs)
    temp = Gram[index]
    beta = np.dot(alpha, temp)
    return beta

def get_gamma(l, Gram, sorted_indices, evecs): 
    '''
    Calculate gamma_l
    '''
    alpha = get_kth_evec(l, sorted_indices, evecs)
    row_sum = np.sum(Gram, axis = 1)
    gamma = np.dot(alpha, row_sum)
    return gamma

def reconstruction_error(index, k, Gram, sorted_indices, evecs): 
    '''
    Calculate reconstruction_error for X[index]
    '''
    N = np.shape(Gram)[0]
    e_1 = Gram[index][index] - 2/N*np.sum(Gram[index]) + 1/(N*N)*np.asmatrix(Gram).sum()

    e_2 = 0
    for i in range(k):
        beta_i = get_beta(index, i, Gram, sorted_indices, evecs)
        gamma_i = get_gamma(i, Gram, sorted_indices, evecs)
        e_2 += np.square((beta_i - gamma_i))

    e_3 = 0
    for l in range(k):
        for m in range(k):
            beta_l = get_beta(index, l, Gram, sorted_indices, evecs) 
            beta_m = get_beta(index, m, Gram, sorted_indices, evecs)
            gamma_l = get_gamma(l, Gram, sorted_indices, evecs)
            gamma_m = get_gamma(m, Gram, sorted_indices, evecs)
            alpha_l = get_kth_evec(l, sorted_indices, evecs)
            alpha_m = get_kth_evec(m, sorted_indices, evecs)
            temp_1 = beta_l*beta_m - 2*beta_l*gamma_m + gamma_l*gamma_m
            dot = np.dot(alpha_m, Gram)
            temp_2 = np.dot(alpha_l, dot)
            e_3 += (temp_1 * temp_2)

    e = e_1 - e_2 + e_3
    return e

def e_rescale(e, mu, sigma): 
    '''
    Rescale the reconstruction_error
    mu: mean, sigma: variance
    '''
    temp = (e - mu) / sigma           
    if temp > 0:
        return temp
    else:
        return 0

def FSVM_N_membership(X, y, sigma_N, threshold, K, **kwargs): 
    '''
    X: data, y: label, K: kernel function, sigma_N: parameter
    '''
    X_pos, X_neg = split(X, y)
    G_pos = G(X_pos, X_pos, **kwargs)
    G_neg = G(X_neg, X_neg, **kwargs)
    n_pos = np.shape(X_pos)[0]
    n_neg = np.shape(X_neg)[0]       
    s_pos = np.zeros(n_pos)
    s_neg = np.zeros(n_neg)
    e_pos = np.zeros(n_pos)
    e_neg = np.zeros(n_neg)

    evals_pos, evecs_pos = np.linalg.eig(G_pos)
    sorted_indices_pos = np.argsort(evals_pos)
    evals_neg, evecs_neg = np.linalg.eig(G_neg)
    sorted_indices_neg = np.argsort(evals_neg)

    k_pos = get_k(threshold, sorted_indices_pos, evals_pos)
    k_neg = get_k(threshold, sorted_indices_neg, evals_neg)
    print(k_neg, k_pos)

    temp = -1/(sigma_N * sigma_N)

    for i in range(n_pos):
        e_pos[i] = reconstruction_error(i, k_pos, G_pos, sorted_indices_pos, evecs_pos)
        print(e_pos[i])
    mu_pos = np.mean(e_pos)
    sigma_pos = np.std(e_pos, ddof = 1)
    for i in range(n_pos):
        s_pos[i] = np.exp(temp * e_rescale(e_pos[i], mu_pos, sigma_pos))

    for j in range(n_neg):
        e_neg[j] = reconstruction_error(j, k_neg, G_neg, sorted_indices_neg, evecs_neg)
        print(e_neg[j])
    mu_neg = np.mean(e_neg)
    sigma_neg = np.std(e_neg, ddof = 1)
    for j in range(n_neg):
        s_neg[j] = np.exp(temp * e_rescale(e_neg[j], mu_neg, sigma_neg))

    s = np.hstack((s_neg, s_pos))
    return s

# 4. Membership based on anormaly detection
def get_gaussian_params(X, useMultivariate):
    '''
    X: row: 样本, column: 特征
    mu: n x 1 vector
    sigma^2: n x 1 vector 或者是(n,n)矩阵, if你使用了多元高斯函数
    '''
    mu = X.mean(axis=0)
    m = np.shape(X)[0]
    if useMultivariate == True:    
        sigma2 = np.dot((X-mu).T, (X-mu)) / m
    else:
        sigma2 = X.var(axis=0, ddof=1)  # 样本方差
    return mu, sigma2

def gaussian_membership(X, mu, sigma2):
    '''
    返回一个(m, )维向量，包含每个样本的概率值

    m, n = np.shape(X) # m: 样本数量, n: 特征数量
    '''
    n = len(mu)
    if np.ndim(sigma2) == 1:
        sigma2 = np.diag(sigma2)
    X = X - mu
    temp = np.linalg.det(sigma2) # 数值过小
    p1 = np.power(np.power(2*np.pi, n)*temp, -1/2)
    print(temp)
    e = np.diag(X.dot(np.linalg.inv(sigma2)).dot(X.T))  # 取对角元素，类似与方差，而不要协方差
    p2 = np.exp(-0.5*e)
    return p1 * p2

def gauss_membership(X, y, useMultivariate):
    X_pos, X_neg = split(X, y)
    cnt = collections.Counter(y)
    n_pos = cnt[1]
    n_neg = cnt[0]
    mu_pos, sigma2_pos = get_gaussian_params(X_pos, useMultivariate)
    mu_neg, sigma2_neg = get_gaussian_params(X_neg, useMultivariate)
    w_pos = gaussian_membership(X_pos, mu_pos, sigma2_pos)
    w_neg = gaussian_membership(X_neg, mu_neg, sigma2_neg)
    w = np.zeros(len(y))
    for i in range(n_pos):
        w[i] = w_pos[i]
    for j in range(n_neg):
        w[n_pos+j] = w_neg[j]
    return w


# My SVDD membership
def QP_solver(Gram, C):
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
    sol = solvers.qp(P, q, G, h, A, b, options={'show_progress': False})
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

def project(d):
    d = np.ravel(d)
    d_min = np.min(d)
    d_max = np.max(d)
    d_avg = (d_max + d_min) / 2
    s = np.zeros(len(d))
    for i in range(len(d)):
        if d[i] > d_avg:
            s[i] = (1 - (d[i] - d_min)/(d_max - d_min))**2 + 0.01
        else:
            s[i] = 1 - (d[i] - d_min)/(d_max - d_min)
    return s


def SVDD_membership(X, y, g, C):
    X_pos, X_neg = split(X, y)
    G_pos = G(X_pos, X_pos, g)
    G_neg = G(X_neg, X_neg, g)
    n_pos = np.shape(X_pos)[0]
    n_neg = np.shape(X_neg)[0]
    try:
        alpha_pos = np.reshape(QP_solver(G_pos, C), (n_pos, 1))
        alpha_neg = np.reshape(QP_solver(G_neg, C), (n_neg, 1))
    except ValueError:
        return np.ones((n_neg + n_pos, 1))
    D_2_pos = np.reshape([get_distance_2(i, G_pos, alpha_pos) for i in range(n_pos)], (n_pos, 1))
    D_2_neg = np.reshape([get_distance_2(i, G_neg, alpha_neg) for i in range(n_neg)], (n_neg, 1))
    d_pos = np.sqrt(D_2_pos)
    d_neg = np.sqrt(D_2_neg)

    d_pos_scale = preprocessing.MinMaxScaler().fit_transform(d_pos)
    d_neg_scale = preprocessing.MinMaxScaler().fit_transform(d_neg)

    s_pos = np.reshape(np.sqrt(np.abs(1 - d_pos_scale)), (n_pos, 1))
    s_neg = np.reshape(np.sqrt(np.abs(1 - d_neg_scale)), (n_neg, 1))
    '''
    d_pos_scale = project(d_pos)
    d_neg_scale = project(d_neg)
    s_pos = np.reshape(d_pos_scale, (n_pos, 1))
    s_neg = np.reshape(d_neg_scale, (n_neg, 1))
    '''
    s = np.row_stack((s_neg, s_pos))
    return s


# SVDD_kernel_precomputed
def SVDD_kernel(X, y, Gram, C):
    X_pos, X_neg = split(X, y)
    n_pos = np.shape(X_pos)[0]
    n_neg = np.shape(X_neg)[0]
    G_neg = Gram[0:n_neg]
    G_neg = G_neg[:, 0:n_neg]
    G_pos = Gram[n_neg:]
    G_pos = G_pos[:, n_neg:]
    try:
        alpha_pos = np.reshape(QP_solver(G_pos, C), (n_pos, 1))
        alpha_neg = np.reshape(QP_solver(G_neg, C), (n_neg, 1))
    except ValueError:
        return []
    D_2_pos = np.reshape([get_distance_2(i, G_pos, alpha_pos) for i in range(n_pos)], (n_pos, 1))
    D_2_neg = np.reshape([get_distance_2(i, G_neg, alpha_neg) for i in range(n_neg)], (n_neg, 1))
    d_pos = np.sqrt(D_2_pos)
    d_neg = np.sqrt(D_2_neg)

    d_pos_scale = preprocessing.MinMaxScaler().fit_transform(d_pos)
    d_neg_scale = preprocessing.MinMaxScaler().fit_transform(d_neg.reshape)

    s_pos = np.reshape(np.sqrt(1 - d_pos_scale), (n_pos, 1))
    s_neg = np.reshape(np.sqrt(1 - d_neg_scale), (n_neg, 1))
    s = np.row_stack((s_neg, s_pos))
    return s

# OneclassSVM 
def OCSVM_membership(X, y, g):
    X_pos, X_neg = split(X, y)

    clf_pos = svm.OneClassSVM(kernel = 'rbf', gamma = g)
    clf_pos.fit(X_pos)
    d_pos = clf_pos.score_samples(X_pos)
    d_pos_scale = preprocessing.MinMaxScaler().fit_transform(d_pos.reshape(-1, 1))

    clf_neg = svm.OneClassSVM(kernel = 'rbf', gamma = g)
    clf_neg.fit(X_neg)
    d_neg = clf_neg.score_samples(X_neg)
    d_neg_scale = preprocessing.MinMaxScaler().fit_transform(d_neg.reshape(-1, 1))

    s_pos = np.sqrt(np.abs(1 - d_pos_scale))
    s_neg = np.sqrt(np.abs(1 - d_neg_scale))

    s = np.row_stack((s_neg, s_pos))
    return s
    
# IFN
def distance_2(index_1, index_2, Gram):
    d_2 = Gram[index_1][index_1] + Gram[index_2][index_2] - 2*Gram[index_1][index_2]
    return d_2

def get_rho(index_i, y_i, Gram, alpha, n_pos, n_neg):
    n_samples = n_pos + n_neg
    if y_i == -1:
        numerator = np.sum([distance_2(index_i, j, Gram) <= alpha**2 for j in range(n_neg, n_samples)] != 0)
    else:
        numerator = np.sum([distance_2(index_i, j, Gram) <= alpha**2 for j in range(n_neg)] != 0)
    denominator = np.sum([distance_2(index_i, j, Gram) <= alpha**2 for j in range(n_samples)] != 0)
    return numerator / denominator

def IFN_membership(X, y, g, C, alpha):
    X_pos, X_neg = split(X, y)
    Gram = G(X, X, g)
    n_pos = np.shape(X_pos)[0]
    n_neg = np.shape(X_neg)[0]
    n_samples = n_pos + n_neg

    mu = np.ravel(SVDD_membership(X, y, g, C))
    rho = np.ravel([get_rho(i, y[i], Gram, alpha, n_pos, n_neg) for i in range(n_samples)])
    nu = np.zeros(n_samples)
    i = 0
    for (mu_i, rho_i) in zip(mu, rho):
        nu[i] = (1 - mu_i) * rho_i
        i = i + 1
    
    s = np.zeros(n_samples)
    i = 0
    for (mu_i, nu_i) in zip(mu, nu):
        if nu_i == 0:
            s[i] = mu_i
        elif mu_i <= nu_i:
            s[i] = 0
        else:
            s[i] = (1 - nu_i) / (2 - mu_i - nu_i)
        i = i + 1
    return s
