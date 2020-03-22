import numpy as np
import collections
from scipy.spatial.distance import cdist
import numba
import pp


@numba.jit(nopython = True, fastmath = True) 
def gaussian(vec1, vec2, g):
    k = np.exp(-g*np.square((np.linalg.norm(vec1 - vec2))))
    return k

def G(X, Y, g):
    K = cdist(X, Y, gaussian, g = g)
    return K

@numba.jit
def split(X, y): 
    '''
    Divide dataset X into two subsets according to their labels y
    '''
    n_pos, n_neg = 0, 0
    for y_i in y:
        if y_i == 1:
            n_pos += 1
        elif y_i == 0:
            n_neg += 1
    n = np.shape(X)[1]
    X_pos = np.zeros((n_pos, n)) 
    X_neg = np.zeros((n_neg, n)) 
    j, k = 0, 0
    for i in range(n_pos + n_neg):
        if y[i] == 0:
            X_neg[j] = X[i]
            j = j + 1
        else:
            X_pos[k] = X[i]
            k = k + 1
    return X_pos, X_neg

# 1. Use Class Center to Reduce the Effects of Outliers
@numba.jit
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
@numba.jit(nopython = True, fastmath = True) 
def get_radius_square(x_i, y_i, X_pos, X_neg, Y, g): 
    '''
    Calculate distance between two points in feature space
    '''
    n_pos = np.shape(X_pos)[0]
    n_neg = np.shape(X_neg)[0]
    k_temp = gaussian(x_i, x_i, g)
    if y_i == 1:
        Gram = G(X_pos, X_pos, g)
        x_i_index = np.where(Gram == x_i)
        temp_1 = np.sum(Gram[x_i_index])
        temp_2 = np.asmatrix(Gram).sum()
        r_square = k_temp - 2/n_pos * temp_1 + 1/(n_pos*n_pos) * temp_2
    elif y_i == 0:
        Gram = G(X_neg, X_neg, g)
        x_i_index = np.where(Gram == x_i)
        temp_1 = np.sum(Gram[x_i_index])
        temp_2 = np.asmatrix(Gram).sum()
        r_square = k_temp - 2/n_neg * temp_1 + 1/(n_neg*n_neg) * temp_2
    return r_square

@numba.jit(nopython = True, fastmath = True, parallel = True) 
def FSVM_2_membership(X, y, delta, g): 
    '''
    X: data, y: label, delta: parameter, K: kernel function
    '''
    X_pos, X_neg = split(X, y)
    n_pos = np.shape(X_pos)[0]
    n_neg = np.shape(X_neg)[0]
    r_square_pos = 0
    r_square_neg = 0
    for i in range(n_pos):
        temp = get_radius_square(X_pos[i], 1, X_pos, X_neg, y, g)
        if temp > r_square_pos:
            r_square_pos = temp
        print(i)
    for j in range(n_neg):
        temp = get_radius_square(X_neg[j], 0, X_pos, X_neg, y, g)
        if temp > r_square_neg:
            r_square_neg = temp
        print(j)
    s = np.zeros(len(y))
    for i in range(len(y)):
        if y[i] == 1:
            d = get_radius_square(X[i], y[i], X_pos, X_neg, y, g)
            s[i] = 1 - np.sqrt(d / (r_square_pos + delta))
        elif y[i] == 0:
            d = get_radius_square(X[i], y[i], X_pos, X_neg, y, g)
            s[i] = 1 - np.sqrt(d / (r_square_neg + delta))
        print(i, s[i])
    return s

# 3. Fuzzy SVM for Noisy Data
@numba.jit
def get_kth_evec(X, k, N, K): 
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

@numba.jit(nopython=True)
def get_beta(x_i, X, N, j, K): 
    '''
    Calculate beta_j
    '''
    alpha = get_kth_evec(X, j, N, K)
    beta = 0
    for i in range(N):
        beta = beta + alpha[i] * K(x_i, X[i])
    return beta

@numba.jit(nopython=True)
def get_gamma(X, N, l, K): 
    '''
    Calculate gamma_l
    '''
    alpha = get_kth_evec(X, l, N, K)
    gamma = 0
    for i in range(N):
        for j in range(N):
            gamma = gamma + alpha[i] * K(X[i], X[j])
    gamma = gamma / N
    return gamma

@numba.jit
def reconstruction_error(x, X, y, k, K): 
    '''
    Calculate reconstruction_error for x
    '''
    N = len(y)
    e_1 = K(x, x) - 2/N*np.sum([K(x, X[i]) for i in range(N)]) + 1/(N*N)*np.sum([K(X[i], X[j]) for i in range(N) for j in range(N)])

    e_2 = 0
    for i in range(k):
        beta_i = get_beta(x, X, N, i, K)
        gamma_i = get_gamma(X, N, i, K)
        e_2 = e_2 + (beta_i*beta_i - 2*beta_i*gamma_i + gamma_i*gamma_i)

    e_3 = 0
    for l in range(k):
        for m in range(k):
            beta_l = get_beta(x, X, N, l, K) 
            beta_m = get_beta(x, X, N, m, K)
            gamma_l = get_gamma(X, N, l, K)
            gamma_m = get_gamma(X, N, m, K)
            alpha_l = get_kth_evec(X, l, N, K)
            alpha_m = get_kth_evec(X, m, N, K)
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

@numba.jit
def FSVM_N_membership(X, y, k, sigma_N, K): 
    '''
    X: data, y: label, K: kernel function, k: PCA dimension, sigma_N: parameter
    '''
    N = len(y)
    s = np.zeros(N)
    e = np.zeros(N)
    for i in range(N):
        e[i] = reconstruction_error(X[i], X, y, k, K)
        print(i, e[i])
    sigma = np.mean(e)
    mu = np.std(e, ddof = 1)
    for i in range(N):
        s[i] = np.exp(-1/(sigma_N*sigma_N) * e_rescale(e[i], mu, sigma))
    return s