import numba
import numpy as np
import membership
from cvxopt import matrix, solvers
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin


@numba.jit(nopython = True) 
def gaussian(vec1, vec2, g):
    k = np.exp(-g*np.square((np.linalg.norm(vec1 - vec2))))
    return k

def Gauss_kernel(X, Y, g):
    K = cdist(X, Y, gaussian, g = g)
    return K

def QP_solver(y, W, Gram, C):
    l = np.shape(Gram)[0]
    y = np.reshape(y, (l, 1))
    W = np.reshape(W, (l, 1))
    P = matrix(np.outer(y, y) * Gram)
    q = matrix(np.ones((l, 1)) * -1)
    i = np.identity(l)
    G = matrix(np.row_stack((-i, i)))
    Zeros = np.zeros((l, 1))
    sC = np.zeros((l, 1))
    for k in range(l):
        sC[k] = C * W[k][0]
    h = matrix(np.row_stack((Zeros, sC)))
    A = matrix(y, (1, l))
    b = matrix(0.0)
    sol = solvers.qp(P, q, G, h, A, b, options={'show_progress': False})
    return sol['x']

class FSVM_Classifier(BaseEstimator, ClassifierMixin):  
    """
    Fuzzy SVM Classifier
    """
    def __init__(self, C = 1, gamma = 0.5, nu = 0.5, membership = 'None', kernel = 'rbf', thres = 0.5):
        self.C = C
        self.gamma = gamma
        self.membership = membership
        self.W = []
        self.kernel = kernel
        self.nu = nu
        self.thres = thres

    def cal_membership(self, X, y):
        n_samples = np.shape(X)[0]
        if self.membership == 'SVDD':
            W = membership.SVDD_membership(X, y, g = self.gamma, C = self.nu)
        elif self.membership == 'None':
            W = np.ones((n_samples, 1))
        elif self.membership == 'OCSVM':
            W = membership.OCSVM_membership(X, y, self.gamma)
        return W

    def fit(self, X, y, Weight = []):
        n_features = np.shape(X)[1]

        if self.membership == 'precomputed':
            W = Weight
        else:
            W = self.cal_membership(X, y)
        
        if self.kernel == 'rbf':
            self.Gram = Gauss_kernel(X, X, self.gamma)
        elif self.kernel == 'linear':
            self.Gram = np.dot(X.T, X)
        elif self.kernel == 'precomputed':
            self.Gram = X
        
        a = np.ravel(QP_solver(y, W, self.Gram, self.C))

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-10
        self.sv_index = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]

        # Intercept b
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * self.Gram[self.sv_index[n], sv])
        self.b /= len(self.a)

        # Weight vector w
        if self.kernel == 'linear':
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None
        
        return self

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for j in range(len(self.a)):
                    if self.kernel == 'rbf':
                        s += self.a[j] * self.sv_y[j] * gaussian(X[i], self.sv[j], self.gamma)
                    elif self.kernel == 'precomputed':
                        s += self.a[j] * self.sv_y[j] * X[i][self.sv_index[j]]
                y_predict[i] = s
            return y_predict + self.b


    def predict(self, X):
        return np.sign(self.project(X))


    #def predict_proba(self, X):


    def decision_function(self, X):
        return self.project(X)

        