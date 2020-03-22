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

def split(X, y): 
    '''
    Divide dataset X into two subsets according to their labels y
    '''
    cnt = collections.Counter(y)
    n_pos = cnt[1] 
    n_neg = cnt[0] 
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
    for j in range(n_pos):
        s_pos[j] = 1 - np.sqrt(get_radius_square(j, X_neg, G_neg, sum_neg, K, **kwargs)/(r_square_neg + delta))
    s = np.hstack((s_neg, s_pos))
    return s

# 3. Fuzzy SVM for Noisy Data
def get_kth_evec(X, k, N, K, **kwargs): 
    '''
    Calculate k_th largest eigenvector
    '''
    G = np.zeros((N, N)) # Kernel matrix
    for i in range(N):
        for j in range(N):
            G[i][j] = K(X[i], X[j], **kwargs)
    evals, evecs = np.linalg.eig(G)
    sorted_indices = np.argsort(evals)
    kth_evec = evecs[:,sorted_indices[-k]]
    return kth_evec

def get_beta(x_i, X, N, j, K, **kwargs): 
    '''
    Calculate beta_j
    '''
    alpha = get_kth_evec(X, j, N, K, **kwargs)
    beta = 0
    for i in range(N):
        beta = beta + alpha[i] * K(x_i, X[i], **kwargs)
    return beta

def get_gamma(X, N, l, K, **kwargs): 
    '''
    Calculate gamma_l
    '''
    alpha = get_kth_evec(X, l, N, K, **kwargs)
    gamma = 0
    for i in range(N):
        for j in range(N):
            gamma = gamma + alpha[i] * K(X[i], X[j], **kwargs)
    gamma = gamma / N
    return gamma

def reconstruction_error(x, X, y, k, K, **kwargs): 
    '''
    Calculate reconstruction_error for x
    '''
    N = len(y)
    e_1 = K(x, x, **kwargs) - 2/N*np.sum([K(x, X[i], **kwargs) for i in range(N)]) + 1/(N*N)*np.sum([K(X[i], X[j], **kwargs) for i in range(N) for j in range(N)])

    e_2 = 0
    for i in range(k):
        beta_i = get_beta(x, X, N, i, K, **kwargs)
        gamma_i = get_gamma(X, N, i, K, **kwargs)
        e_2 = e_2 + (beta_i*beta_i - 2*beta_i*gamma_i + gamma_i*gamma_i)

    e_3 = 0
    for l in range(k):
        for m in range(k):
            beta_l = get_beta(x, X, N, l, K, **kwargs) 
            beta_m = get_beta(x, X, N, m, K, **kwargs)
            gamma_l = get_gamma(X, N, l, K, **kwargs)
            gamma_m = get_gamma(X, N, m, K, **kwargs)
            alpha_l = get_kth_evec(X, l, N, K, **kwargs)
            alpha_m = get_kth_evec(X, m, N, K, **kwargs)
            temp = beta_l*beta_m - 2*beta_l*gamma_m + gamma_l*gamma_m
            for i in range(N):
                for j in range(N):
                    e_3 = e_3 + temp * alpha_l[i] * alpha_m[j] * K(X[i], X[j], **kwargs)
    e = e_1 + e_2 + e_3
    return e

def store_error(file_path, X, y, k, K, **kwargs):
    N = len(y)
    mat = np.zeros(N)
    for i in range(N):
        mat[i] = reconstruction_error(X[i], X, y, k, K, **kwargs)
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in mat:
            writer.writerow(row)
    csvfile.close()
    return mat

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

def FSVM_N_membership(X, y, k, sigma_N, K, **kwargs): 
    '''
    X: data, y: label, K: kernel function, k: PCA dimension, sigma_N: parameter
    '''
    N = len(y)
    s = np.zeros(N)
    e = np.zeros(N)
    for i in range(N):
        e[i] = reconstruction_error(X[i], X, y, k, K, **kwargs)
        print(i, e[i])
    sigma = np.mean(e)
    mu = np.std(e, ddof = 1)
    for i in range(N):
        s[i] = np.exp(-1/(sigma_N*sigma_N) * e_rescale(e[i], mu, sigma))
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
    if np.ndim(sigma2) == 1:
        sigma2 = np.diag(sigma2)
    norm = 1./(np.power((2*np.pi), n/2)*np.sqrt(np.linalg.det(sigma2)))
    exp = np.zeros((m,1))
    for row in range(m):
        xrow = X[row]
        exp[row] = np.exp(-0.5*((xrow-mu).T).dot(np.linalg.inv(sigma2)).dot(xrow-mu))
    return norm*exp
    '''
    n = len(mu)
    if np.ndim(sigma2) == 1:
        sigma2 = np.diag(sigma2)
    X = X - mu
    p1 = np.power(2*np.pi, -n/2)*np.power(np.linalg.det(sigma2), -1/2)
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