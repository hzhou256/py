import sys
path='D:/Program Files/libsvm_weights-3.23/python'
sys.path.append(path)
import numpy as np
import membership
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
    for it in range(1,2):
        name = methods_name[it]
        print(name + ':')

        f1 = np.loadtxt('E:/Study/Bioinformatics/FuzzySVM/feature_matrix/' + name_ds +'/' + name + '/train_' + name + '.csv', delimiter = ',', skiprows = 1)
        f2 = np.loadtxt('E:/Study/Bioinformatics/FuzzySVM/feature_matrix/' + name_ds + '/train_label.csv', delimiter = ',')
        X_train = get_feature(f1)
        y_train = f2

        y, X = svm_read_problem('E:/Study/Bioinformatics/FuzzySVM/feature_matrix/' + name_ds +'/' + name + '/train_' + name + '.svm')
        y_test, X_test = svm_read_problem('E:/Study/Bioinformatics/FuzzySVM/feature_matrix/' + name_ds +'/' + name + '/test_' + name + '.svm')
        W = membership.FSVM_2_membership(X_train, y_train, 0.1, membership.tanimoto)
        print(W)
        '''
        prob = svm_problem(W, y, X)

        cv = model_selection.StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
        #space = {'kernel':['rbf'],'C':np.logspace(-7, 3, base = 10), 'gamma':np.logspace(-7, 5, base = 10)}
        #grid = model_selection.GridSearchCV(svm.SVC(kernel = 'rbf', probability = True), space, n_jobs = -1, cv = cv)
        #grid.fit(X_train, y_train)
        #C = grid.best_params_['C']
        #g = grid.best_params_['gamma']

        # create a function to minimize.
        def SVM_accuracy_cv(params, cv = cv, X = X_train, y = y_train):
            # the function gets a set of variable parameters in "param"
            params = {'C': params['C'], 'gamma': params['gamma']}
            # we use this params to create a new LinearSVC Classifier
            model = svm.SVC(kernel = 'rbf', probability = True, **params)
            # and then conduct the cross validation with the same folds as before
            score = -model_selection.cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs = -1).mean()
            return score

        # possible values of parameters
        space= {'C': hp.loguniform('C', low = np.log(1e-7) , high = np.log(1e2)), 
                'gamma': hp.loguniform('gamma', low = np.log(1e-7) , high = np.log(1e5))}

        # trials will contain logging information
        trials = Trials()
        best = fmin(fn = SVM_accuracy_cv, # function to optimize
                space = space,
                algo = tpe.suggest, # optimization algorithm, hyperotp will select its parameters automatically
                max_evals = 50, # maximum number of iterations
                trials = trials, # logging
                )
        C = best['C']
        g = best['gamma']
        print('C =', C)
        print('g =', g)

        
        C = 1.4
        g = 77

        param = svm_parameter('-t 2 -c '+str(C)+' -g '+str(g)+' -b 1')
        m = svm_train(prob, param)
        #CV_ACC = svm_train(W, y, X, '-v 5')
        p_label, p_acc, p_val = svm_predict(y_test, X_test, m)
        print(p_acc[0])
        '''