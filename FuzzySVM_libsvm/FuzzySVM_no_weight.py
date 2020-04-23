import sys
path='D:/Program Files/libsvm_weights-3.23/python'
sys.path.append(path)
import csv
import numpy as np
import membership
import anomaly_detection
from sklearn import svm
from svmutil import *
from sklearn import model_selection
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


def get_feature(file):
    m = np.shape(file)[0]
    n = np.shape(file)[1]
    data = np.zeros((m, n-1))
    for index in range(m):
        data[index] = file[index][1:]
    return data

dataset_name = ['Antifp_Main', 'Antifp_DS1', 'Antifp_DS2']
for ds in range(1):
    name_ds = dataset_name[ds]
    print('dataset:', name_ds)
    methods_name = ['188-bit', 'AAC', 'ASDC', 'CKSAAP', 'CTD', 'DPC']
    for it in range(5,6):
        name = methods_name[it]
        print(name + ':')

        f1 = np.loadtxt('E:/Study/Bioinformatics/FuzzySVM/feature_matrix/' + name_ds +'/' + name + '/train_' + name + '.csv', delimiter = ',', skiprows = 1)
        f2 = np.loadtxt('E:/Study/Bioinformatics/FuzzySVM/feature_matrix/' + name_ds + '/train_label.csv', delimiter = ',')
        X_train = get_feature(f1)
        y_train = f2

        f3 = np.loadtxt('E:/Study/Bioinformatics/FuzzySVM/feature_matrix/' + name_ds +'/' + name + '/test_' + name + '.csv', delimiter = ',', skiprows = 1)
        f4 = np.loadtxt('E:/Study/Bioinformatics/FuzzySVM/feature_matrix/' + name_ds + '/test_label.csv', delimiter = ',')
        X_test = get_feature(f3)
        y_test = f4

        y_svm, X_svm = svm_read_problem('E:/Study/Bioinformatics/FuzzySVM/feature_matrix/' + name_ds +'/' + name + '/train_' + name + '.svm')
        y_test_svm, X_test_svm = svm_read_problem('E:/Study/Bioinformatics/FuzzySVM/feature_matrix/' + name_ds +'/' + name + '/test_' + name + '.svm')

        # create a function to minimize.
        def svm_weight_ACC(params, X_svm = X_svm, y_svm = y_svm, X = X_train, y = y_train):
            params = {'C': params['C'], 'gamma': params['gamma']}
            W = []
            prob = svm_problem(W, y_svm, X_svm)
            param = svm_parameter('-t 2 -c '+str(params['C'])+' -g '+str(params['gamma'])+' -v 5')
            score = svm_train(prob, param)
            return -score

        # possible values of parameters
        space = {'C': hp.loguniform('C', low = np.log(1e-7) , high = np.log(1e2)), 
                'gamma': hp.loguniform('gamma', low = np.log(1e-7) , high = np.log(1e5))}

        # trials will contain logging information
        trials = Trials()
        best = fmin(fn = svm_weight_ACC, # function to optimize
                space = space,
                algo = tpe.suggest, # optimization algorithm, hyperotp will select its parameters automatically
                max_evals = 100, # maximum number of iterations
                trials = trials, # logging
                )
        C = best['C']
        g = best['gamma']
        print('C =', C)
        print('g =', g)

        W = []
        prob = svm_problem(W, y_svm, X_svm)
        param = svm_parameter('-t 2 -c '+str(C)+' -g '+str(g))
        m = svm_train(prob, param)
        p_label, p_acc, p_val = svm_predict(y_test_svm, X_test_svm, m)
        print(p_acc[0])
