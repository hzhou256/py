import sys
path='D:/Program Files/libsvm_weights-3.23/python'
sys.path.append(path)
import numpy as np
import membership
from sklearn import preprocessing
from svmutil import *
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn import metrics
from imblearn.metrics import specificity_score


delta = 0.001

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


y_svm, X_svm = svm_read_problem('E:/Study/Bioinformatics/UCI/' + name + '/X_train.svm')
y_test_svm, X_test_svm = svm_read_problem('E:/Study/Bioinformatics/UCI/' + name + '/X_test.svm')


def svm_weight_ACC(params, X = X_svm, y = y_svm, W = []):
    params = {'C': params['C'], 'gamma': params['gamma']}
    prob = svm_problem(W, y, X)
    param = svm_parameter('-t 2 -c '+str(params['C'])+' -g '+str(params['gamma'])+' -v 10')
    score = svm_train(prob, param)
    return -score

def svm_weight_ACC_2(params, X_svm = X_svm, y_svm = y_svm, X = X_train, y = y_train):
    params = {'C': params['C'], 'gamma': params['gamma']}
    W = membership.FSVM_2_membership(X, y, delta, membership.gaussian, g = params['gamma'])
    #print(W)
    prob = svm_problem(W, y_svm, X_svm)
    param = svm_parameter('-t 2 -c '+str(params['C'])+' -g '+str(params['gamma'])+' -v 10')
    score = svm_train(prob, param)
    return -score

def svm_weight_ACC_N(params, X_svm = X_svm, y_svm = y_svm, X = X_train, y = y_train):
    params = {'C': params['C'], 'gamma': params['gamma'], 'sigmaN': params['sigmaN']}
    W = membership.FSVM_N_membership(X, y, params['sigmaN'], 0.90, membership.gaussian, g = params['gamma'])
    print(W)
    prob = svm_problem(W, y_svm, X_svm)
    param = svm_parameter('-t 2 -c '+str(params['C'])+' -g '+str(params['gamma'])+' -v 10')
    score = svm_train(prob, param)
    return -score

space = {'C': hp.loguniform('C', low = np.log(1e-7) , high = np.log(1e3)), 'gamma': hp.loguniform('gamma', low = np.log(1e-7) , high = np.log(1e5))}

space_N = {'C': hp.loguniform('C', low = np.log(1e-7) , high = np.log(1e3)), 
        'gamma': hp.loguniform('gamma', low = np.log(1e-7) , high = np.log(1e5)), 'sigmaN': hp.loguniform('sigmaN', low = np.log(1e-3) , high = np.log(1e3))}

trials = Trials()
best = fmin(fn = svm_weight_ACC_N, 
        space = space_N,
        algo = tpe.suggest, 
        max_evals = 100, 
        trials = trials,
        )
C = best['C']
g = best['gamma']

#W = []
#W = membership.FSVM_2_membership(X_train, y_train, delta, membership.gaussian, g = g)  
W = membership.FSVM_N_membership(X_train, y_train, sigmaN, 0.85, membership.gaussian, g = g)
prob = svm_problem(W, y_svm, X_svm)
param = svm_parameter('-t 2 -c '+str(C)+' -g '+str(g) + ' -b 1')
m = svm_train(prob, param)
p_label, p_acc, p_val = svm_predict(y_test_svm, X_test_svm, m, '-b 1')
y_prob = np.reshape([p_val[i][0] for i in range(np.shape(p_val)[0])], (np.shape(p_val)[0], 1))

ACC = metrics.accuracy_score(y_test, p_label)
precision = metrics.precision_score(y_test, p_label)
sensitivity = metrics.recall_score(y_test, p_label)
specificity = specificity_score(y_test, p_label)
AUC = metrics.roc_auc_score(y_test, y_prob)
MCC = metrics.matthews_corrcoef(y_test, p_label)

print('C =', C)
print('g =', g)
#print('sigmaN =', sigmaN)

print('SN =', sensitivity)
print('SP =', specificity)
print('ACC =', p_acc[0])
print('MCC =', MCC)
print('AUC =', AUC)
