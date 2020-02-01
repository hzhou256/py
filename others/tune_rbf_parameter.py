import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection
from imblearn.metrics import specificity_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from read import read_file_np1
from read import read_label
from sklearn.multiclass import OneVsRestClassifier

methods_name = ['gene', 'isoform', 'meth']

gamma_new = []
C_new = []
for it in range(3):
    name = methods_name[it]
    print(name)
    sample = read_file_np1('D:/下载/LUNG/' + 'LUNG_' + name + '_train' + '.csv')
    label = read_label('D:/下载/LUNG/' + 'LUNG_' + name + '_train_label' + '.csv')

    print(sample.T.shape)
    print(label.shape)
    cv = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


    # parameters = {'C': np.logspace(-5, 2, base = 10)}
    # grid = model_selection.GridSearchCV(svm.SVC(kernel = 'precomputed', probability = True), parameters, n_jobs = -1, cv = cv)
    # grid.fit(gram, y_train)
    # cost = grid.best_params_['C']
    # print('C =', cost)

    # create a function to minimize.
    def SVM_accuracy_cv(params, cv=cv, X=sample.T, y=label):
        # the function gets a set of variable parameters in "param"
        params = {'C': params['C'], 'gamma': params['gamma']}
        # we use this params to create a new LinearSVC Classifier
        model = svm.SVC(kernel='rbf', probability=True, **params)
        # and then conduct the cross validation with the same folds as before
        score = -model_selection.cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1).mean()
        return score


    # possible values of parameters
    space= {'C': hp.loguniform('C', low = np.log(1e-7) , high = np.log(1e3)), 
                'gamma': hp.loguniform('gamma', low = np.log(1e-7) , high = np.log(1e5))}

    # trials will contain logging information
    trials = Trials()
    best = fmin(fn=SVM_accuracy_cv,  # function to optimize
                space=space,
                algo=tpe.suggest,  # optimization algorithm, hyperotp will select its parameters automatically
                max_evals=100,  # maximum number of iterations
                trials=trials,  # logging
                )
    cost = best['C']
    gamma1 = best['gamma']
    gamma_new.append(gamma1)
    C_new.append(cost)
    print('C =', cost)
    print('gamma',gamma1)

print(gamma_new)
print(C_new)
