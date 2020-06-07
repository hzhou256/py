import numpy as np
import membership, Fuzzy_SVM
from sklearn import metrics
from imblearn.metrics import specificity_score
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold
from sklearn import svm


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

dataset = ['australian', 'breastw', 'diabetes', 'german', 'heart', 'ionosphere', 'sonar', 'imbalance']
for i in range(0, 7):
    name = dataset[i]
    print(name)
    f1 = np.loadtxt('E:/Study/Bioinformatics/UCI/' + name + '/data.csv', delimiter = ',')
    X = f1[:, 0:-1]
    y = f1[:, -1]


    X_train, y_train = split(X, y)

    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
    #parameters = {'C': np.logspace(-10, 10, base = 2, num = 21), 'gamma': np.logspace(5, -15, base = 2, num = 21), 'nu': np.linspace(0.1, 0.5, num = 5)}
    #parameters = {'C': np.logspace(-10, 10, base = 2, num = 21), 'gamma': np.logspace(5, -15, base = 2, num = 21), 'nu': [0.1]}
    parameters = {'C': np.logspace(-10, 10, base = 2, num = 21), 'gamma': np.logspace(5, -15, base = 2, num = 21)}
    #grid = GridSearchCV(Fuzzy_SVM.FSVM_Classifier(membership = 'SVDD', proj = 'normal'), parameters, n_jobs = -1, cv = cv, verbose = 1)
    #grid = GridSearchCV(svm.SVC(kernel = 'rbf'), parameters, n_jobs = -1, cv = cv, verbose = 1)
    grid = GridSearchCV(Fuzzy_SVM.FSVM_Classifier(membership = 'FSVM_2'), parameters, n_jobs = -1, cv = cv, verbose = 1)
    grid.fit(X_train, y_train)
    gamma = grid.best_params_['gamma']
    C = grid.best_params_['C']
    #nu = grid.best_params_['nu']

    #clf = Fuzzy_SVM.FSVM_Classifier(C = C, gamma = gamma, membership = 'SVDD', nu = nu, proj = 'normal')
    #clf = svm.SVC(C = C, gamma = gamma, kernel = 'rbf', probability = True)
    clf = Fuzzy_SVM.FSVM_Classifier(C = C, gamma = gamma, membership = 'FSVM_2')
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
    mean_AP = np.mean(five_fold['test_AP'])

    print(mean_sensitivity)
    print(mean_SP)
    print(mean_ACC)
    print(mean_MCC)
    print(mean_AUC)
    print(mean_AUPR)
    #print(mean_AP)

    print('C = ', C)
    print('g = ', gamma)
    #print('nu = ', nu)