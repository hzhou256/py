import numpy as np
from sklearn import metrics
from imblearn.metrics import specificity_score
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier


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

methods_name = ['ASDC', 'DPC', 'CKSAAP']
for it in range(0, 3):
    name = methods_name[it]
    print(name + ':')
    f1 = np.loadtxt('E:/Study/Bioinformatics/DeepAVP/'+name+'/train_'+name+'.csv', delimiter = ',')
    X_train = f1[:, 0:-1]
    y_train = f1[:, -1]
    X_train, y_train = split(X_train, y_train)

    f2 = np.loadtxt('E:/Study/Bioinformatics/DeepAVP/'+name+'/test_'+name+'.csv', delimiter = ',')
    X_test = f2[:, 0:-1]
    y_test = f2[:, -1]

    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)

    parameters_1 = {'n_estimators': np.arange(1, 200)}
    parameters_2 = {'max_depth': np.arange(1, 100), 'criterion': ['gini', 'entropy']}

    grid_1 = GridSearchCV(RandomForestClassifier(), parameters_1, n_jobs = -1, cv = cv, verbose = 1)
    grid_1.fit(X_train, y_train)

    n_estimators = grid_1.best_params_['n_estimators']

    grid_2 = GridSearchCV(RandomForestClassifier(n_estimators = n_estimators), parameters_2, n_jobs = -1, cv = cv, verbose = 1)
    grid_2.fit(X_train, y_train)

    max_depth = grid_2.best_params_['max_depth']
    criterion = grid_2.best_params_['criterion']

    clf = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, criterion = criterion)
    clf.fit(X_train, y_train)

    scorerMCC = metrics.make_scorer(metrics.matthews_corrcoef)
    scorerSP = metrics.make_scorer(specificity_score)
    scorerPR = metrics.make_scorer(metrics.precision_score)
    scorerSE = metrics.make_scorer(metrics.recall_score)
    scoreAUPR = metrics.make_scorer(get_AUPR, needs_threshold = True)
    scorer = {'ACC':'accuracy', 'recall':scorerSE, 'roc_auc':'roc_auc', 'MCC':scorerMCC, 'SP':scorerSP, 'AUPR':scoreAUPR}

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

    print('n_estimators = ', n_estimators)
    print('max_depth = ', max_depth)
    print('criterion = ', criterion)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    ACC = metrics.accuracy_score(y_test, y_pred)
    sensitivity = metrics.recall_score(y_test, y_pred)
    specificity = specificity_score(y_test, y_pred)
    AUC = metrics.roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
    MCC = metrics.matthews_corrcoef(y_test, y_pred)
    AUPR = get_AUPR(y_test, clf.predict_proba(X_test)[:,1])

    print('Testing set:')
    print(sensitivity)
    print(specificity)
    print(ACC)
    print(MCC)
    print(AUC)
    print(AUPR)