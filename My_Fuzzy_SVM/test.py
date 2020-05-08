import numpy as np
import membership, Fuzzy_SVM
from sklearn import metrics
from imblearn.metrics import specificity_score
from sklearn.model_selection import cross_validate, StratifiedKFold


def Resplit(X, y): 
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

C_list = [0.1110, 0.84446, 3.2655, 1024.0, 730.2239,  6.4216, 12.6278]
g_list = [0.006824, 0.0048664, 3.05176e-05, 0.000118, 3.05176e-05, 0.2814, 0.553379]
alpha_list = np.linspace(0, 2, num = 100)

dataset = ['australian', 'breastw', 'diabetes', 'german', 'heart', 'ionosphere', 'sonar']
for i in range(6, 7):
    name = dataset[i]
    print(name)
    f1 = np.loadtxt('E:/Study/Bioinformatics/UCI/' + name + '/data.csv', delimiter = ',')
    X = f1[:, 0:-1]
    y = f1[:, -1]

    X_train, y_train = Resplit(X, y)

    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)

    C = C_list[i]
    gamma = g_list[i]

    for a in alpha_list:
        alpha = a
        clf = Fuzzy_SVM.FSVM_Classifier(C = C, gamma = gamma, membership = 'IFN_SVDD', alpha = alpha)
        clf.fit(X_train, y_train)

        scorerMCC = metrics.make_scorer(metrics.matthews_corrcoef)
        scorerSP = metrics.make_scorer(specificity_score)
        scorerPR = metrics.make_scorer(metrics.precision_score)
        scorerSE = metrics.make_scorer(metrics.recall_score)
        scorer = {'ACC':'accuracy', 'recall':scorerSE, 'roc_auc': 'roc_auc', 'MCC':scorerMCC, 'SP':scorerSP}

        five_fold = cross_validate(clf, X_train, y_train, cv = cv, scoring = scorer, verbose = 0, n_jobs = 4)

        mean_ACC = np.mean(five_fold['test_ACC'])
        mean_sensitivity = np.mean(five_fold['test_recall'])
        mean_AUC = np.mean(five_fold['test_roc_auc'])
        mean_MCC = np.mean(five_fold['test_MCC'])
        mean_SP = np.mean(five_fold['test_SP'])

        #print(mean_sensitivity)
        #print(mean_SP)
        print(mean_ACC)
        #print(mean_MCC)
        #print(mean_AUC)
