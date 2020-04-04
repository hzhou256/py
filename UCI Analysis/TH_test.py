import sys
path='D:/Program Files/libsvm_weights-3.23/python'
sys.path.append(path)
import numpy as np
import membership
import SVDD
import TH
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

def svm_weight_ACC_TH(params, X_svm = X_svm, y_svm = y_svm, X = X_train, y = y_train):
    params = {'C': params['C'], 'C_1': params['C_1'], 'C_2': params['C_2'], 'gamma': params['gamma'], 'v_1': params['v_1'], 'v_2': params['v_2']}
    W = TH.TH_membership(X, y, g = params['gamma'], C_1 = params['C_1'], C_2 = params['C_2'], v_1 = params['v_1'], v_2 = params['v_2'])
    print(W)
    prob = svm_problem(W, y_svm, X_svm)
    param = svm_parameter('-t 2 -c '+str(params['C'])+' -g '+str(params['gamma'])+' -v 10')
    score = svm_train(prob, param)
    return -score

space_TH = {'C': hp.loguniform('C', low = np.log(1e-3) , high = np.log(1e3)), 'C_1': hp.loguniform('C_1', low = np.log(1e-3) , high = np.log(1e3)), 
        'C_2': hp.loguniform('C_2', low = np.log(1e-3) , high = np.log(1e3)), 'gamma': hp.loguniform('gamma', low = np.log(1e-5) , high = np.log(1e5)), 
        'v_1': hp.loguniform('v_1', low = np.log(1e-3) , high = np.log(1e0)), 'v_2': hp.loguniform('v_2', low = np.log(1e-3) , high = np.log(1e0))}

trials = Trials()
best = fmin(fn = svm_weight_ACC_TH, 
        space = space_TH,
        algo = tpe.suggest, 
        max_evals = 100, 
        trials = trials,
        )

C = best['C']
C_1 = best['C_1']
C_2 = best['C_2']
g = best['gamma']
v_1 = best['v_1']
v_2 = best['v_2']

#W = []
W = TH.TH_membership(X_train, y_train, g, C_1, C_2, v_1, v_2)
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
print('C_1 =', C_1)
print('C_2 =', C_2)
print('g =', g)
print('v_1 =', v_1)
print('v_2 =', v_2)

#print('SN =', sensitivity)
#print('SP =', specificity)
#print('ACC =', p_acc[0])
#print('MCC =', MCC)
#print('AUC =', AUC)
