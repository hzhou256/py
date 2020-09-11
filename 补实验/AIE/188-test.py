import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection
from scipy.spatial.distance import cdist
from imblearn.metrics import specificity_score


def get_y_score(y_proba):
    n = np.shape(y_proba)[0]
    temp = np.zeros(n)
    for i in range(n):
        temp[i] = y_proba[i][1]
    return temp

def get_AUPR(y_true, y_score):
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_score, pos_label = 1)
    AUPR = metrics.auc(recall, precision)
    return AUPR


methods_name = ['AAC', 'ASDC', 'CKSAAP', 'DPC', '188-bit']
for it in range(4, 5):
    name = methods_name[it]

    gram_train = np.loadtxt('D:/Study/Bioinformatics/补实验/AIE/kernels/K_train_'+name+'.csv', delimiter = ',')
    gram_test = np.loadtxt('D:/Study/Bioinformatics/补实验/AIE/kernels/K_test_'+name+'.csv', delimiter = ',')

    y_train = np.loadtxt('D:/Study/Bioinformatics/补实验/AIE/features/train_label.csv', delimiter = ',')
    y_test = np.loadtxt('D:/Study/Bioinformatics/补实验/AIE/features/test_label.csv', delimiter = ',')


    cv = model_selection.StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)

    parameters = {'C': np.logspace(-15, 10, base = 2, num = 52)}
    grid = model_selection.GridSearchCV(svm.SVC(kernel = 'precomputed', probability = True), parameters, n_jobs = -1, cv = cv, verbose = 2)
    grid.fit(gram_train, y_train)
    C = grid.best_params_['C']
    print('C =', C)


    clf = svm.SVC(C = C, kernel = 'precomputed', probability = True)

    scorerMCC = metrics.make_scorer(metrics.matthews_corrcoef)
    scorerSP = metrics.make_scorer(specificity_score)
    scorerPR = metrics.make_scorer(metrics.precision_score)
    scorerSE = metrics.make_scorer(metrics.recall_score)

    scorer = {'ACC':'accuracy', 'recall':scorerSE, 'roc_auc': 'roc_auc', 'MCC':scorerMCC, 'SP':scorerSP}
    five_fold = model_selection.cross_validate(clf, gram_train, y_train, cv = cv, scoring = scorer)

    mean_ACC = np.mean(five_fold['test_ACC'])
    mean_sensitivity = np.mean(five_fold['test_recall'])
    mean_AUC = np.mean(five_fold['test_roc_auc'])
    mean_MCC = np.mean(five_fold['test_MCC'])
    mean_SP = np.mean(five_fold['test_SP'])

    #print('five fold:')
    print(mean_sensitivity)
    print(mean_SP)
    print(mean_ACC)
    print(mean_MCC)
    print(mean_AUC)


    clf.fit(gram_train, y_train)

    y_score = clf.predict_proba(gram_test)
    y_score = get_y_score(y_score)
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_score)

    y_pred = clf.predict(gram_test)
    ACC = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    sensitivity = metrics.recall_score(y_test, y_pred)
    specificity = specificity_score(y_test, y_pred)
    AUC = metrics.roc_auc_score(y_test, y_score)
    MCC = metrics.matthews_corrcoef(y_test, y_pred)
    AUPR = get_AUPR(y_test, y_score)

    print("===========================")
    #print('testing:')
    print(sensitivity)
    print(specificity)
    print(ACC)
    print(MCC)
    print(AUC)
    #print('AUPR =', AUPR)

