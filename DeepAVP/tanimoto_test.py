import numpy as np
import membership, Fuzzy_SVM
from sklearn import metrics
from imblearn.metrics import specificity_score
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold
from sklearn import svm


def get_AUPR(y_true, y_score):
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_score, pos_label = 1)
    AUPR = metrics.auc(recall, precision)
    return AUPR


y_train = np.loadtxt('E:/Study/Bioinformatics/DeepAVP/y_train.csv', delimiter = ',')
y_test = np.loadtxt('E:/Study/Bioinformatics/DeepAVP/y_test.csv', delimiter = ',')

methods_name = ['ASDC', 'CKSAAP', 'DPC']
for it in range(0, 3):
    name = methods_name[it]
    print(name + ':')
    gram_train = np.loadtxt('E:/Study/Bioinformatics/DeepAVP/' + name + '/gram_train_' + name + '.csv', delimiter = ',')
    gram_test = np.loadtxt('E:/Study/Bioinformatics/DeepAVP/' + name + '/gram_test_' + name + '.csv', delimiter = ',')

    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
    parameters = {'C': np.logspace(-10, 10, base = 2, num = 42)}

    grid = GridSearchCV(svm.SVC(kernel = 'precomputed'), parameters, n_jobs = -1, cv = cv, verbose = 1)
    grid.fit(gram_train, y_train)
    C = grid.best_params_['C']

    clf = svm.SVC(C = C, kernel = 'precomputed', probability = True)

    scorerMCC = metrics.make_scorer(metrics.matthews_corrcoef)
    scorerSP = metrics.make_scorer(specificity_score)
    scorerPR = metrics.make_scorer(metrics.precision_score)
    scorerSE = metrics.make_scorer(metrics.recall_score)
    scoreAUPR = metrics.make_scorer(get_AUPR, needs_threshold = True)
    scorer = {'ACC':'accuracy', 'recall':scorerSE, 'roc_auc':'roc_auc', 'MCC':scorerMCC, 'SP':scorerSP, 'AUPR':scoreAUPR}
    five_fold = cross_validate(clf, gram_train, y_train, cv = cv, scoring = scorer, n_jobs = -1)

    mean_ACC = np.mean(five_fold['test_ACC'])
    mean_sensitivity = np.mean(five_fold['test_recall'])
    mean_AUC = np.mean(five_fold['test_roc_auc'])
    mean_MCC = np.mean(five_fold['test_MCC'])
    mean_SP = np.mean(five_fold['test_SP'])
    mean_AUPR = np.mean(five_fold['test_AUPR'])

    print('five fold:')
    print(mean_sensitivity)
    print(mean_SP)
    print(mean_ACC)
    print(mean_MCC)
    print(mean_AUC)
    print(mean_AUPR)

    print('C = ', C)

    clf.fit(gram_train, y_train)
    y_pred = clf.predict(gram_test)
    ACC = metrics.accuracy_score(y_test, y_pred)
    sensitivity = metrics.recall_score(y_test, y_pred)
    specificity = specificity_score(y_test, y_pred)
    AUC = metrics.roc_auc_score(y_test, clf.decision_function(gram_test))
    MCC = metrics.matthews_corrcoef(y_test, y_pred)
    AUPR = get_AUPR(y_test, clf.decision_function(gram_test))

    print('Testing set:')
    print(sensitivity)
    print(specificity)
    print(ACC)
    print(MCC)
    print(AUC)
    print(AUPR)