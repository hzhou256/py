import pandas as pd
import numpy as np


def gaussian(X, mu, sigma2):
    '''
    mu, sigma2参数已经决定了一个高斯分布模型
    因为原始模型就是多元高斯模型在sigma2上是对角矩阵而已，所以如下：
    If Sigma2 is a matrix, it is treated as the covariance matrix. 
    If Sigma2 is a vector, it is treated as the sigma^2 values of the variances
    in each dimension (a diagonal covariance matrix)
    output:
        一个(m, )维向量，包含每个样本的概率值。
    '''
# 如果想用矩阵相乘求解exp()中的项，一定要注意维度的变换。
# 事实上我们只需要取对角线上的元素即可。（类似于方差而不是想要协方差）
# 最后得到一个（m，）的向量，包含每个样本的概率，而不是想要一个（m,m）的矩阵
# 注意这里，当矩阵过大时，numpy矩阵相乘会出现内存错误。例如9万维的矩阵。所以画图时不能生成太多数据~！
#     n = len(mu) 
    
#     if np.ndim(sigma2) == 1:
#         sigma2 = np.diag(sigma2)

#     X = X - mu
#     p1 = np.power(2 * np.pi, -n/2)*np.sqrt(np.linalg.det(sigma2))
#     e = np.diag(X@np.linalg.inv(sigma2)@X.T)  # 取对角元素，类似与方差，而不要协方差
#     p2 = np.exp(-.5*e)
    
#     return p1 * p2

# 下面是不利用矩阵的解法，相当于把每行数据输入进去，不会出现内存错误。
    m, n = X.shape # m: 样本数量, n: 特征数量
    if np.ndim(sigma2) == 1:
        sigma2 = np.diag(sigma2)

    norm = 1./(np.power((2*np.pi), n/2)*np.sqrt(np.linalg.det(sigma2)))
    exp = np.zeros((m,1))
    for row in range(m):
        xrow = X[row]
        exp[row] = np.exp(-0.5*((xrow-mu).T).dot(np.linalg.inv(sigma2)).dot(xrow-mu))
    return norm*exp

def getGaussianParams(X, useMultivariate):
    """
    The input X is: row: 样本, column: 特征
    The output is an n-dimensional vector mu, the mean of the data set 
    the variances sigma^2, an n x 1 vector 或者是(n,n)矩阵，if你使用了多元高斯函数
    """
    mu = X.mean(axis=0)
    if useMultivariate == True:    
        sigma2 = np.dot((X-mu).T, (X-mu)) / len(X)
    else:
        sigma2 = X.var(axis=0, ddof=1)  # 样本方差
    
    return mu, sigma2


