#import os
#os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from sklearn import svm
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
methods_name = ['188-bit', 'AAC', 'ASDC', 'CKSAAP', 'CTD', 'DPC']

for ds in range(1,2):
    name_ds = dataset_name[ds]
    print(name_ds)
    for it in range(2,3):
        name = methods_name[it]
        print(name)
        f1 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds + '/' + name +'/train_' + name +'.csv', delimiter = ',', skiprows = 1)
        f2 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds + '/train_label.csv', delimiter = ',')
        f3 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds + '/' + name +'/test_' + name +'.csv', delimiter = ',', skiprows = 1)
        f4 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds + '/test_label.csv', delimiter = ',')

        np.set_printoptions(suppress = True)
        X_train = get_feature(f1)
        y_train = f2
        X_test = get_feature(f3)
        y_test = f4

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

        # possible values of parameters
        space= {'C': hp.loguniform('C', low = np.log(1e-7) , high = np.log(1e2)), 
                'gamma': hp.loguniform('gamma', low = np.log(1e-7) , high = np.log(1e5))}

        # trials will contain logging information
        trials = Trials()
        best = fmin(fn = SVM_accuracy_cv, # function to optimize
                space = space,
                algo = tpe.suggest, # optimization algorithm, hyperotp will select its parameters automatically
                max_evals = 100, # maximum number of iterations
                trials = trials, # logging
                )
        print(best)
