import sys
path = 'D:/Program Files/libsvm_weights-3.23/python'
sys.path.append(path)
from svmutil import *
import numpy as np
import My_Fuzzy_SVM
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn import metrics, preprocessing, svm
from imblearn.metrics import specificity_score
import membership
import collections

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

g = 0.00024
s = membership.OCSVM_membership(X_train, y_train, g = g)

C = 128
prob = svm_problem(W = s, y = y_train, x = X_train)
param = svm_parameter('-t 2 -c '+str(C)+' -g '+str(g) + ' -b 1')
m = svm_train(prob, param)
p_label, p_acc, p_val = svm_predict(y_test, X_test, m, '-b 1')