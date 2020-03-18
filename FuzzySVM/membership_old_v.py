import numpy
import collections
import numba
import pp


@numba.jit
def gaussian(vec1, vec2, g):
    k = numpy.exp(-g*numpy.square((numpy.linalg.norm(vec1 - vec2))))
    return k

@numba.jit
def tanimoto(p_vec, q_vec):
    pq = numpy.dot(p_vec, q_vec)
    p_square = numpy.square(numpy.linalg.norm(p_vec))
    q_square = numpy.square(numpy.linalg.norm(q_vec))
    d = pq / (p_square + q_square - pq)
    return d

def split(X, y): 
    '''
    Divide dataset X into two subsets according to their labels y
    '''
    cnt = collections.Counter(y)
    n_pos = cnt[1] 
    n_neg = cnt[0] 
    n = numpy.shape(X)[1]
    X_pos = numpy.zeros((n_pos, n)) 
    X_neg = numpy.zeros((n_neg, n)) 
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
    mean_pos = numpy.mean(X_pos, axis = 0)
    mean_neg = numpy.mean(X_neg, axis = 0)
    temp_pos = numpy.zeros(n_pos)
    temp_neg = numpy.zeros(n_neg)
    for i in range(n_pos):
        temp_pos[i] = numpy.linalg.norm(mean_pos - X_pos[i])
    for i in range(n_neg):
        temp_neg[i] = numpy.linalg.norm(mean_neg - X_neg[i])
    r_pos = numpy.max(temp_pos)
    r_neg = numpy.max(temp_neg)
    s = numpy.zeros(len(y))
    for i in range(len(y)):
        if y[i] == 1:
            s[i] = 1 - numpy.linalg.norm(mean_pos - X[i])/(r_pos + delta)
        elif y[i] == 0:
            s[i] = 1 - numpy.linalg.norm(mean_neg - X[i])/(r_neg + delta)
    return s

# 2. Fuzzy membership function for nonlinear SVM
@numba.jit
def get_radius_square(x_i, y_i, X, Y, K): 
    '''
    Calculate distance between two points in feature space
    '''
    cnt = collections.Counter(Y)
    n_pos = cnt[1]
    n_neg = cnt[0]
    X_pos, X_neg = split(X, Y)

    k_temp = K(x_i, x_i)
    if y_i == 1:
        temp_1, temp_2 = 0, 0
        for i in range(n_pos):
            temp_1 = temp_1 + K(x_i, X_pos[i])
            for j in range(n_pos):
                temp_2 = temp_2 + K(X_pos[i], X_pos[j])
        r_square = k_temp - 2/n_pos * temp_1 + 1/(n_pos*n_pos) * temp_2
    elif y_i == 0:
        temp_1, temp_2 = 0, 0
        for i in range(n_neg):
            temp_1 = temp_1 + K(x_i, X_neg[i])
            for j in range(n_neg):
                temp_2 = temp_2 + K(X_neg[i], X_neg[j])
        r_square = k_temp - 2/n_neg * temp_1 + 1/(n_neg*n_neg) * temp_2
    print(r_square)
    return r_square

@numba.jit
def FSVM_2_membership(X, y, delta, K): 
    '''
    X: data, y: label, delta: parameter, K: kernel function
    '''
    X_pos, X_neg = split(X, y)
    r_square_pos = numpy.max([(get_radius_square(x, 1, X, y, K)) for x in X_pos])
    r_square_neg = numpy.max([(get_radius_square(x, 0, X, y, K)) for x in X_neg])
    s = numpy.zeros(len(y))
    for i in range(len(y)):
        if y[i] == 1:
            s[i] = 1 - numpy.sqrt(numpy.linalg.norm(get_radius_square(X[i], y[i], X, y, K))/(r_square_pos + delta))
        elif y[i] == 0:
            s[i] = 1 - numpy.sqrt(numpy.linalg.norm(get_radius_square(X[i], y[i], X, y, K))/(r_square_neg + delta))
        print(i, s[i])
    return s

# 3. Fuzzy SVM for Noisy Data
@numba.jit
def get_kth_evec(X, k, N, K): 
    '''
    Calculate k_th largest eigenvector
    '''
    G = numpy.zeros((N, N)) # Kernel matrix
    for i in range(N):
        for j in range(N):
            G[i][j] = K(X[i], X[j])
    evals, evecs = numpy.linalg.eig(G)
    sorted_indices = numpy.argsort(evals)
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
    e_1 = K(x, x) - 2/N*numpy.sum([K(x, X[i]) for i in range(N)]) + 1/(N*N)*numpy.sum([K(X[i], X[j]) for i in range(N) for j in range(N)])

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
    s = numpy.zeros(N)
    e = numpy.zeros(N)
    for i in range(N):
        e[i] = reconstruction_error(X[i], X, y, k, K)
        print(i, e[i])
    sigma = numpy.mean(e)
    mu = numpy.std(e, ddof = 1)
    for i in range(N):
        s[i] = numpy.exp(-1/(sigma_N*sigma_N) * e_rescale(e[i], mu, sigma))
    return s