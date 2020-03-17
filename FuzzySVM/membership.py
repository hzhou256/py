import numpy as np
from collections import Counter


def split(X, y): 
    '''
    Divide dataset X into two subsets according to their labels y
    '''
    cnt = Counter(y)
    n_pos = cnt[1] 
    n_neg = cnt[-1] 
    X_pos = np.zeros((n_pos, 2)) 
    X_neg = np.zeros((n_neg, 2)) 
    j, k = 0, 0
    for i in range(n_pos + n_neg):
        if y[i] == -1:
            X_neg[j] = X[i]
            j = j + 1
        else:
            X_pos[k] = X[i]
            k = k + 1
    return X_pos, X_neg

# 1. Use Class Center to Reduce the Effects of Outliers
def class_center_membership(X, y, delta): 
    '''
    X: data, y: label, delta: parameter
    '''
    X_pos, X_neg = split(X, y)
    mean_pos = np.mean(X_pos)
    mean_neg = np.mean(X_neg)
    r_pos = np.max([np.linalg.norm(mean_pos - x) for x in X_pos])
    r_neg = np.max([np.linalg.norm(mean_neg - x) for x in X_neg])
    s = np.zeros(len(y))
    for i in range(len(y)):
        if y[i] == 1:
            s[i] = 1 - np.linalg.norm(mean_pos - X[i])/(r_pos + delta)
        elif y[i] == -1:
            s[i] = 1 - np.linalg.norm(mean_neg - X[i])/(r_neg + delta)
    return s

# 2. Fuzzy membership function for nonlinear SVM
def get_radius_square(x_i, y_i, X, Y, K): 
    '''
    Calculate distance between two points in feature space
    '''
    cnt = Counter(Y)
    n_pos = cnt[1]
    n_neg = cnt[-1]
    X_pos, X_neg = split(X, Y)
    if y_i == 1:
        r_square = K(x_i, x_i) - 2/n_pos * np.sum([(K(x_i, x_j)) for x_j in X_pos])\
             + 1/(n_pos*n_pos) * np.sum([(K(X_pos[i], X_pos[j])) for i in range(n_pos) for j in range(n_pos)])
    elif y_i == -1:
        r_square = K(x_i, x_i) - 2/n_neg * np.sum([(K(x_i, x_j)) for x_j in X_neg])\
             + 1/(n_neg*n_neg) * np.sum([(K(X_neg[i], X_neg[j])) for i in range(n_neg) for j in range(n_neg)])
    return r_square

def FSVM_2_membership(X, y, delta, K): 
    '''
    X: data, y: label, delta: parameter, K: kernel function
    '''
    X_pos, X_neg = split(X, y)
    r_square_pos = np.max([(get_radius_square(x, 1, X, y, K)) for x in X_pos])
    r_square_neg = np.max([(get_radius_square(x, -1, X, y, K)) for x in X_neg])
    s = np.zeros(len(y))
    for i in range(len(y)):
        if y[i] == 1:
            s[i] = 1 - np.sqrt(np.linalg.norm(get_radius_square(X[i], y[i], X, y, K))/(r_square_pos + delta))
        elif y[i] == -1:
            s[i] = 1 - np.sqrt(np.linalg.norm(get_radius_square(X[i], y[i], X, y, K))/(r_square_neg + delta))
    return s

# 3. Fuzzy SVM for Noisy Data
def get_kth_evec(X, K, k, N): 
    '''
    Calculate k_th largest eigenvector
    '''
    G = np.zeros((N, N)) # Kernel matrix
    for i in range(N):
        for j in range(N):
            G[i][j] = K(X[i], X[j])
    evals, evecs = np.linalg.eig(G)
    sorted_indices = np.argsort(evals)
    kth_evec = evecs[:,sorted_indices[-k]]
    return kth_evec

def get_beta(x_i, X, K, N, j): 
    '''
    Calculate beta_j
    '''
    alpha = get_kth_evec(X, K, j, N)
    beta = 0
    for i in range(N):
        beta = beta + alpha[i] * K(x_i, X[i])
    return beta

def get_gamma(X, K, N, l): 
    '''
    Calculate gamma_l
    '''
    alpha = get_kth_evec(X, K, l, N)
    gamma = 0
    for i in range(N):
        for j in range(N):
            gamma = gamma + alpha[i] * K(X[i], X[j])
    gamma = gamma / N
    return gamma

def reconstruction_error(x, X, y, K, k): 
    '''
    Calculate reconstruction_error for x
    '''
    N = len(y)
    e_1 = K(x, x) - 2/N*np.sum([K(x, X[i]) for i in range(N)]) + 1/(N*N)*np.sum([K(X[i], X[j]) for i in range(N) for j in range(N)])

    e_2 = 0
    for i in range(k):
        beta_i = get_beta(x, X, K, N, i)
        gamma_i = get_gamma(X, K, N, i)
        e_2 = e_2 + (beta_i*beta_i - 2*beta_i*gamma_i + gamma_i*gamma_i)

    e_3 = 0
    for l in range(k):
        for m in range(k):
            beta_l = get_beta(x, X, K, N, l) 
            beta_m = get_beta(x, X, K, N, m)
            gamma_l = get_gamma(X, K, N, l)
            gamma_m = get_gamma(X, K, N, m)
            alpha_l = get_kth_evec(X, K, l, N)
            alpha_m = get_kth_evec(X, K, m, N)
            temp = beta_l*beta_m - 2*beta_l*gamma_m + gamma_l*gamma_m
            for i in range(N):
                for j in range(N):
                    e_3 = e_3 + temp * alpha_l[i] * alpha_m[j] * K(X[i], X[j])
    e = e_1 + e_2 + e_3
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

def FSVM_N_membership(X, y, K, k, sigma_N): 
    '''
    X: data, y: label, K: kernel function, k: PCA dimension, sigma_N: parameter
    '''
    N = len(y)
    s = np.zeros(N)
    e = np.zeros(N)
    for i in range(N):
        e[i] = reconstruction_error(X[i], X, y, K, k)
    sigma = np.mean(e)
    mu = np.std(e, ddof = 1)
    for i in range(N):
        s[i] = np.exp(-1/(sigma_N*sigma_N) * e_rescale(e[i], mu, sigma))
    return s