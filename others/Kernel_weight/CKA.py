import math
import numpy as np
import metrics_function as mf


def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n

    return np.dot(np.dot(H, K), H)  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
    # return np.dot(H, K)  # KH


#def rbf(X, sigma=None):
#    GX = np.dot(X, X.T)
#    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
#    if sigma is None:
#        mdist = np.median(KX[KX != 0])
#        sigma = math.sqrt(mdist)
#    KX *= - 0.5 / (sigma * sigma)
#    KX = np.exp(KX)
#    return KX


#def kernel_HSIC(X, Y, sigma):
#    return np.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))


#def linear_HSIC(X, Y):
#    L_X = np.dot(X, X.T)
#    L_Y = np.dot(Y, Y.T)
#    return np.sum(centering(L_X) * centering(L_Y))


#def linear_CKA(X, Y):
#    hsic = linear_HSIC(X, Y)
#    var1 = np.sqrt(linear_HSIC(X, X))
#    var2 = np.sqrt(linear_HSIC(Y, Y))

#    return hsic / (var1 * var2)


#def kernel_CKA(X, Y, sigma=None):
#    hsic = kernel_HSIC(X, Y, sigma)
#    var1 = np.sqrt(kernel_HSIC(X, X, sigma))
#    var2 = np.sqrt(kernel_HSIC(Y, Y, sigma))

#    return hsic / (var1 * var2)

def my_HSIC(X, Y):
    if X.ndim == 1:
        temp = X
        t1 = temp.reshape(len(temp), 1)
        t2 = temp.reshape(1, len(temp))
        T_X = np.dot(t1, t2)
    else:
        T_X = mf.tanimoto(X, X)    
    if Y.ndim == 1:
        temp = Y
        t1 = temp.reshape(len(temp), 1)
        t2 = temp.reshape(1, len(temp))
        T_Y = np.dot(t1, t2)
    else:
        T_Y = mf.tanimoto(Y, Y)
    return np.sum(centering(T_X) * centering(T_Y))

def my_CKA(X, Y):
    hsic = my_HSIC(X, Y)
    var1 = np.sqrt(my_HSIC(X, X))
    var2 = np.sqrt(my_HSIC(Y, Y))
    return hsic / (var1 * var2)

def get_feature(file):
    m = np.shape(file)[0]
    n = np.shape(file)[1]
    data = np.zeros((m, n-1))
    for index in range(m):
        data[index] = file[index][1:]
    return data

np.set_printoptions(suppress = True)
dataset_name = ['Antifp_Main', 'Antifp_DS1', 'Antifp_DS2']
for ds in range(3):
    name_ds = dataset_name[ds]
    print('dataset:', name_ds)
    methods_name = ['188-bit', 'AAC', 'ASDC', 'CKSAAP', 'CTD', 'DPC']
    for it in range(6):
        name = methods_name[it]
        print(name + ':')
        file = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds +'/' + name + '/train_' + name + '.csv', delimiter = ',', skiprows = 1)
        X = get_feature(file)
        Y = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds + '/train_label.csv')
        weight = my_CKA(X, Y)
        print(weight)
