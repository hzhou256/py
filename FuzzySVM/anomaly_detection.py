import pandas as pd
import numpy as np


def gaussian(X, mu, sigma2):
    '''
    返回一个(m, )维向量，包含每个样本的概率值
    '''
    m, n = np.shape(X) # m: 样本数量, n: 特征数量
    if np.ndim(sigma2) == 1:
        sigma2 = np.diag(sigma2)
    norm = 1./(np.power((2*np.pi), n/2)*np.sqrt(np.linalg.det(sigma2)))
    exp = np.zeros((m,1))
    for row in range(m):
        xrow = X[row]
        exp[row] = np.exp(-0.5*((xrow-mu).T).dot(np.linalg.inv(sigma2)).dot(xrow-mu))
    return norm*exp

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


