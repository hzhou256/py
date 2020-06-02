import numpy as np
import membership, Fuzzy_SVM
from sklearn import metrics
from imblearn.metrics import specificity_score
from sklearn.model_selection import cross_validate, StratifiedKFold, cross_val_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


def split(X, y): 
    '''
    y = {-1, 1}
    '''
    n_pos, n_neg = 0, 0
    for y_i in y:
        if y_i == 1:
            n_pos += 1
        elif y_i == -1:
            n_neg += 1
    n = np.shape(X)[1]
    X_pos = np.zeros((n_pos, n)) 
    X_neg = np.zeros((n_neg, n)) 
    j, k = 0, 0
    for i in range(n_pos + n_neg):
        if y[i] == -1:
            X_neg[j] = X[i]
            j = j + 1
        else:
            X_pos[k] = X[i]
            k = k + 1
    y = np.zeros(n_neg + n_pos)
    for i in range(n_neg):
        y[i] = -1
    for i in range(n_neg, n_neg + n_pos):
        y[i] = 1
    X = np.row_stack((X_neg, X_pos))
    return X, y

dataset = ['australian', 'breastw', 'diabetes', 'german', 'heart', 'ionosphere', 'sonar']
for i in range(2, 3):
    name = dataset[i]
    print(name)
    f1 = np.loadtxt('E:/Study/Bioinformatics/UCI/' + name + '/data.csv', delimiter = ',')
    X = f1[:, 0:-1]
    y = f1[:, -1]

    X_train, y_train = split(X, y)

    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)

    def SVM_accuracy_cv(params, cv = cv, X = X_train, y = y_train):
        params = {'C': params['C'], 'gamma': params['gamma'], 'nu': params['nu']}
        model = Fuzzy_SVM.FSVM_Classifier(membership = 'SVDD', proj = 'beta', kernel = 'rbf')
        score = -cross_val_score(model, X, y, cv = cv, scoring = "accuracy", n_jobs = -1).mean()
        return score

    space= {'C': hp.uniform('C', low = 2**-10 , high = 2**10), 'gamma': hp.uniform('gamma', low = 2**-15 , high = 2**5), 'nu': hp.uniform('nu', low = 0, high = 1)}

    trials = Trials()
    best = fmin(fn = SVM_accuracy_cv, 
            space = space,
            algo = tpe.suggest, 
            max_evals = 200,
            trials = trials, 
            )
    C = best['C']
    g = best['gamma']
    nu = best['nu']
    print('C =', C)
    print('gamma =', g)
    print('nu =', nu)

    clf = Fuzzy_SVM.FSVM_Classifier(C = C, gamma = g, membership = 'SVDD', nu = nu, proj = 'beta')
    clf.fit(X_train, y_train)

    scorerMCC = metrics.make_scorer(metrics.matthews_corrcoef)
    scorerSP = metrics.make_scorer(specificity_score)
    scorerPR = metrics.make_scorer(metrics.precision_score)
    scorerSE = metrics.make_scorer(metrics.recall_score)
    scorer = {'ACC':'accuracy', 'recall':scorerSE, 'roc_auc': 'roc_auc', 'MCC':scorerMCC, 'SP':scorerSP}

    five_fold = cross_validate(clf, X_train, y_train, cv = cv, scoring = scorer, n_jobs = -1)
 
    mean_ACC = np.mean(five_fold['test_ACC'])
    mean_sensitivity = np.mean(five_fold['test_recall'])
    mean_AUC = np.mean(five_fold['test_roc_auc'])
    mean_MCC = np.mean(five_fold['test_MCC'])
    mean_SP = np.mean(five_fold['test_SP'])

    print(mean_sensitivity)
    print(mean_SP)
    print(mean_ACC)
    print(mean_MCC)
    print(mean_AUC)