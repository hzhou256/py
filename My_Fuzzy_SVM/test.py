import numpy as np
import membership, Fuzzy_SVM
from sklearn import metrics
from imblearn.metrics import specificity_score
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV


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

def get_AUPR(y_true, y_score):
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_score, pos_label = 1)
    AUPR = metrics.auc(recall, precision)
    return AUPR


dataset = ['australian', 'breastw', 'diabetes', 'german', 'heart', 'ionosphere', 'sonar', 'mushroom', 'bupa',  'transfusion', 'spam']
for i in range(2, 3):
    name = dataset[i]
    print(name)
    f1 = np.loadtxt('E:/Study/Bioinformatics/UCI/' + name + '/data.csv', delimiter = ',')

    X = f1[:, 0:-1]
    y = f1[:, -1]

    X_train, y_train = split(X, y)

    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)

    parameters = {'nu': np.linspace(0.1, 1, num = 10), 'gamma': np.logspace(5, -15, base = 2, num = 21)}
    grid = GridSearchCV(Fuzzy_SVM.FSVM_Classifier(membership = 'SVDD', proj = 'beta'), parameters, n_jobs = -1, cv = cv, verbose = 1)
    grid.fit(X_train, y_train)
    nu = grid.best_params_['nu']
    g = grid.best_params_['gamma']

    parameters_2 = {'C': np.logspace(-10, 10, base = 2, num = 21)}
    grid_2 = GridSearchCV(Fuzzy_SVM.FSVM_Classifier(membership = 'SVDD', proj = 'beta', gamma = g, nu = nu), parameters_2, n_jobs = -1, cv = cv, verbose = 1)
    grid_2.fit(X_train, y_train)
    C = grid_2.best_params_['C']

    clf = Fuzzy_SVM.FSVM_Classifier(C = C, gamma = g, membership = 'SVDD', nu = nu, proj = 'beta')
    clf.fit(X_train, y_train)

    scorerMCC = metrics.make_scorer(metrics.matthews_corrcoef)
    scorerSP = metrics.make_scorer(specificity_score)
    scorerPR = metrics.make_scorer(metrics.precision_score)
    scorerSE = metrics.make_scorer(metrics.recall_score)
    scoreAUPR = metrics.make_scorer(get_AUPR, needs_threshold = True)
    scorer = {'ACC':'accuracy', 'recall':scorerSE, 'roc_auc':'roc_auc', 'MCC':scorerMCC, 'SP':scorerSP, 'AUPR':scoreAUPR, 'AP': 'average_precision'}

    five_fold = cross_validate(clf, X_train, y_train, cv = cv, scoring = scorer, n_jobs = -1)

    mean_ACC = np.mean(five_fold['test_ACC'])
    mean_sensitivity = np.mean(five_fold['test_recall'])
    mean_AUC = np.mean(five_fold['test_roc_auc'])
    mean_MCC = np.mean(five_fold['test_MCC'])
    mean_SP = np.mean(five_fold['test_SP'])
    mean_AUPR = np.mean(five_fold['test_AUPR'])

    print(mean_sensitivity)
    print(mean_SP)
    print(mean_ACC)
    print(mean_MCC)
    print(mean_AUC)
    print(mean_AUPR)

    print('C = ', C)
    print('g = ', g)
    print('nu = ', nu)
