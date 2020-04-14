import sys
path='D:/Program Files/libsvm_weights-3.23/python'
sys.path.append(path)
import numpy as np
import membership
import SVDD
from sklearn import preprocessing
from svmutil import *
from hyperopt import fmin, tpe, rand, hp, STATUS_OK, Trials
from sklearn import metrics
from imblearn.metrics import specificity_score


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

#y_svm, X_svm = svm_read_problem('E:/Study/Bioinformatics/UCI/' + name + '/X_train.svm')
#y_test_svm, X_test_svm = svm_read_problem('E:/Study/Bioinformatics/UCI/' + name + '/X_test.svm')
y_svm = y_train
y_test_svm = y_test
X_svm = X_train
X_test_svm = X_test


def svm_weight_ACC(params, X = X_svm, y = y_svm, W = []):
    params = {'C': params['C'], 'gamma': params['gamma']}
    prob = svm_problem(W = W, y = y, x = X)
    param = svm_parameter('-t 2 -c '+str(params['C'])+' -g '+str(params['gamma'])+' -v 10')
    score = svm_train(prob, param)
    return -score

def svm_weight_ACC_SVDD(params, X_svm = X_svm, y_svm = y_svm, X = X_train, y = y_train):
    params = {'C': params['C'], 'gamma': params['gamma']}
    W = SVDD.SVDD_membership(X, y, g = params['gamma'], C = params['C'])
    #print(W)
    prob = svm_problem(W = W, y = y_svm, x = X_svm)
    param = svm_parameter('-t 2 -c '+str(params['C'])+' -g '+str(params['gamma'])+' -v 10')
    score = svm_train(prob, param)
    return -score

space_SVDD = {'C': hp.loguniform('C', low = np.log(1e-3) , high = np.log(1e3)), 'gamma': hp.loguniform('gamma', low = np.log(1e-5) , high = np.log(1e5))}

trials = Trials()
best = fmin(fn = svm_weight_ACC_SVDD, 
        space = space_SVDD,
        algo = tpe.suggest, 
        max_evals = 100, 
        trials = trials,
        )

C = best['C']
g = best['gamma']

#W = []
W = SVDD.SVDD_membership(X_train, y_train, g, C)
prob = svm_problem(W = W, y = y_svm, x = X_svm)
param = svm_parameter('-t 2 -c '+str(C)+' -g '+str(g) + ' -b 1')
m = svm_train(prob, param)
p_label, p_acc, p_val = svm_predict(y_test_svm, X_test_svm, m, '-b 1')


print('C =', C)
print('g =', g)

