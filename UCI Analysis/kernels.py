import sys
path='D:/Program Files/libsvm_weights-3.23/python'
sys.path.append(path)
import numpy as np
import numba
import SVDD_kernel_precomputed
from svmutil import *
from scipy.spatial.distance import cdist
from cvxopt import matrix, solvers
from sklearn.model_selection import GridSearchCV
from sklearn import svm, model_selection, preprocessing
from hyperopt import fmin, tpe, rand, hp, STATUS_OK, Trials


@numba.jit(nopython = True, fastmath = True) 
def gaussian(vec1, vec2, g):
    k = np.exp(-g*np.square((np.linalg.norm(vec1 - vec2))))
    return k

def Gauss(X, Y, g):
    K = cdist(X, Y, gaussian, g = g)
    return K

def Poly(X, Y, gamma, r, degree):
    temp = gamma * np.dot(X, Y) + r
    return np.power(temp, degree)

def Sigmoid(X, Y, gamma, r):
    temp = gamma * np.dot(X, Y) + r
    return np.tanh(temp)

def get_feature(file):
    m = np.shape(file)[0]
    n = np.shape(file)[1]
    data = np.zeros((m, n-1))
    for index in range(m):
        data[index] = file[index][1:]
    return data


dataset = ['ionosphere', 'german']
name = dataset[1]

f1 = np.loadtxt('E:/Study/Bioinformatics/UCI/' + name + '/X_train.csv', delimiter = ',', skiprows = 1)
X_train = get_feature(f1)
y_train = f1[:, 0]

f2 = np.loadtxt('E:/Study/Bioinformatics/UCI/' + name + '/X_test.csv', delimiter = ',', skiprows = 1)
X_test = get_feature(f2)
y_test = f2[:, 0]

scaler = preprocessing.MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

index_column_train = np.zeros((np.shape(X_train)[0], 1))
for i in range(np.shape(X_train)[0]):
    index_column_train[i] = i + 1

index_column_test = np.zeros((np.shape(X_test)[0], 1))
for j in range(np.shape(X_test)[0]):
    index_column_test[j] = j + 1

# Gaussian kernel
'''
parameters = {'C': np.logspace(-10, 10, base = 2), 'gamma': np.logspace(5, -5, base = 10)}
clf = GridSearchCV(svm.SVC(), parameters, n_jobs = -1, cv = 5, verbose = 1)
clf.fit(X_train, y_train)
gamma = clf.best_params_['gamma']
C = clf.best_params_['C']
print('Gauss')
print('C = ', C)
print('gamma = ', gamma)
print(clf.best_score_)
'''
gamma = 0.029470517025518096
K_train_gauss = Gauss(X_train, X_train, gamma)
K_test_gauss = Gauss(X_test, X_train, gamma)

# Linear kernel
K_train_linear = np.dot(X_train, X_train.T)
K_test_linear = np.dot(X_test, X_train.T)

# Polynomial kernel
gamma = 0.0167683293681101
r = 3.7275937203149416
degree = 5
K_train_poly = Poly(X_train, X_train.T, gamma, r, degree)
K_test_poly = Poly(X_test, X_train.T, gamma, r, degree)

K_train = (K_train_gauss + K_train_linear + K_train_poly) / 3
K_test = (K_test_gauss + K_test_linear + K_test_poly) / 3
K_train_SVM = np.column_stack((index_column_train, K_train))
K_test_SVM = np.column_stack((index_column_test, K_test))

def svm_weight_ACC(params, X = X_train, y = y_train, K_svm = K_train_SVM):
    params = {'C': params['C']}
    W = SVDD_kernel_precomputed.SVDD_membership(X_train, y_train, K_train, params['C'])
    #W = []
    prob = svm_problem(y = y_train, x = K_svm, W = W, isKernel = True)
    param = svm_parameter('-t 4 -c '+str(params['C'])+' -v 5')
    score = svm_train(prob, param)
    return -score
space = {'C': hp.loguniform('C', low = np.log(1e-5), high = np.log(1e3))}
trials = Trials()
best = fmin(fn = svm_weight_ACC,
        space = space,
        algo = tpe.suggest,
        max_evals = 100,
        trials = trials,
        )
C = best['C']

W = SVDD_kernel_precomputed.SVDD_membership(X_train, y_train, K_train, C)
#W = []
prob  = svm_problem(y = y_train, x = K_train_SVM, W = W, isKernel = True)
param = svm_parameter('-t 4 -c '+str(C)+' -b 1')
m = svm_train(prob, param)
print(C)
p_label, p_acc, p_val = svm_predict(y_test, K_test_SVM, m, '-b 1')
