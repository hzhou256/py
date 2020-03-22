import sys
path='D:/Program Files/libsvm_weights-3.23/python'
sys.path.append(path)
import csv
import numpy as np
import membership
import membership_old_v
import anomaly_detection
from sklearn import svm
from svmutil import *
from sklearn import model_selection
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import collections


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

        cv = model_selection.StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)

        # create a function to minimize.
        def SVM_accuracy_cv(params, cv = cv, X = X_train, y = y_train):
            # the function gets a set of variable parameters in "param"
            params = {'C': params['C'], 'gamma': params['gamma']}
            # we use this params to create a new LinearSVC Classifier
            model = svm.SVC(kernel = 'rbf', probability = True, **params)
            # and then conduct the cross validation with the same folds as before
            score = -model_selection.cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs = -1).mean()
            return score

        def svm_weight_ACC_1(params, X_svm = X_svm, y_svm = y_svm, X = X_train, y = y_train):
            params = {'C': params['C'], 'gamma': params['gamma'], 'delta': params['delta']}
            W = membership.class_center_membership(X, y, params['delta'])
            print(W)
            prob = svm_problem(W, y_svm, X_svm)
            param = svm_parameter('-t 2 -c '+str(params['C'])+' -g '+str(params['gamma'])+' -v 5')
            score = svm_train(prob, param)
            return -score
        
        def svm_weight_ACC_2(params, X_svm = X_svm, y_svm = y_svm, X = X_train, y = y_train):
            params = {'C': params['C'], 'gamma': params['gamma'], 'delta': params['delta']}
            W = membership.FSVM_2_membership(X, y, params['delta'], membership.gaussian, g = params['gamma'])
            prob = svm_problem(W, y_svm, X_svm)
            param = svm_parameter('-t 2 -c '+str(params['C'])+' -g '+str(params['gamma'])+' -v 5')
            score = svm_train(prob, param)
            return -score

        def svm_weight_ACC_N(params, X_svm = X_svm, y_svm = y_svm, X = X_train, y = y_train):
            params = {'C': params['C'], 'gamma': params['gamma'], 'sigmaN': params['sigmaN']}
            W = membership.FSVM_N_membership(X, y, 2, params['sigmaN'], membership.gaussian, g = params['gamma'])
            prob = svm_problem(W, y_svm, X_svm)
            param = svm_parameter('-t 2 -c '+str(params['C'])+' -g '+str(params['gamma'])+' -v 5')
            score = svm_train(prob, param)
            return -score
        
        def svm_weight_ACC_gauss(params, X_svm = X_svm, y_svm = y_svm, X = X_train, y = y_train):
            params = {'C': params['C'], 'gamma': params['gamma']}
            W = membership.gauss_membership(X, y, False)
            cnt = collections.Counter(W)
            print(cnt)
            prob = svm_problem(W, y_svm, X_svm)
            param = svm_parameter('-t 2 -c '+str(params['C'])+' -g '+str(params['gamma'])+' -v 5')
            score = svm_train(prob, param)
            return -score

        # possible values of parameters
        space_1 = {'C': hp.loguniform('C', low = np.log(1e-7) , high = np.log(1e3)), 
                'gamma': hp.loguniform('gamma', low = np.log(1e-7) , high = np.log(1e5)), 'delta': hp.loguniform('delta', low = np.log(1e-3) , high = np.log(1e3))}

        space_2 = {'C': hp.loguniform('C', low = np.log(1e-7) , high = np.log(1e3)), 
                'gamma': hp.loguniform('gamma', low = np.log(1e-7) , high = np.log(1e5)), 'delta': hp.loguniform('delta', low = np.log(1e-3) , high = np.log(1e3))}

        space_N = {'C': hp.loguniform('C', low = np.log(1e-7) , high = np.log(1e3)), 
                'gamma': hp.loguniform('gamma', low = np.log(1e-7) , high = np.log(1e5)), 'sigmaN': hp.loguniform('sigmaN', low = np.log(1e-3) , high = np.log(1e3))}

        space_gauss = {'C': hp.loguniform('C', low = np.log(1e-7) , high = np.log(1e3)), 'gamma': hp.loguniform('gamma', low = np.log(1e-7) , high = np.log(1e5))}

        # trials will contain logging information
        trials = Trials()
        best = fmin(fn = svm_weight_ACC_2, # function to optimize
                space = space_2,
                algo = tpe.suggest, # optimization algorithm, hyperotp will select its parameters automatically
                max_evals = 100, # maximum number of iterations
                trials = trials, # logging
                )
        C = best['C']
        g = best['gamma']
        delta = best['delta']
        #sigmaN = best['sigmaN']
        print('C =', C)
        print('g =', g)
        print('delta =', delta)
        #print('sigmaN =', sigmaN)

        #W = membership.class_center_membership(X_train, y_train, delta)
        W = membership.FSVM_2_membership(X_train, y_train, delta, membership.gaussian, g = g)
        #with open('E:/Study/Bioinformatics/FuzzySVM/feature_matrix/' + name_ds +'/' + name + '/W_FSVM_2_' + name + '.csv', 'w', newline='') as csvfile:
        #    writer = csv.writer(csvfile)
        #    for row in W:
        #        writer.writerow(row)
        #csvfile.close()

        #W = membership.gauss_membership(X_train, y_train, False)
        #W = membership.FSVM_N_membership(X_train, y_train, 2, sigmaN, membership.gaussian, g = g)
        prob = svm_problem(W, y_svm, X_svm)
        param = svm_parameter('-t 2 -c '+str(C)+' -g '+str(g))
        m = svm_train(prob, param)
        #CV_ACC = svm_train(W, y_svm, X_svm, '-v 5')
        p_label, p_acc, p_val = svm_predict(y_test_svm, X_test_svm, m)
        print(p_acc[0])
