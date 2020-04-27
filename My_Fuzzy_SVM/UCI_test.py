import numpy as np
import membership, Fuzzy_SVM
from sklearn import metrics
from imblearn.metrics import specificity_score
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split, StratifiedKFold


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

dataset = ['australian', 'heart', 'sonar']
name = dataset[0]

f1 = np.loadtxt('E:/Study/Bioinformatics/UCI/' + name + '/data.csv', delimiter = ',')
X = f1[:, 0:-1]
y = f1[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_train, y_train = split(X_train, y_train)

nu_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
parameters = {'C': np.logspace(-10, 10, base = 2, num = 42), 'gamma': np.logspace(5, -15, base = 2, num = 42)}

best_mean_ACC = 0
best_mean_sensitivity = 0
best_mean_AUC = 0
best_mean_MCC = 0
best_mean_SP = 0
best_ACC = 0
best_sensitivity = 0
best_specificity = 0
best_AUC = 0
best_MCC = 0
best_C = 0
best_gamma = 0
best_nu = 0

for i in range(len(nu_list)):
#for i in range(1):
    nu = nu_list[i]
    #nu = 0.5
    grid = GridSearchCV(Fuzzy_SVM.FSVM_Classifier(membership = 'SVDD', nu = nu), parameters, n_jobs = -1, cv = cv, verbose = 1)
    grid.fit(X_train, y_train)
    gamma = grid.best_params_['gamma']
    C = grid.best_params_['C']

    clf = Fuzzy_SVM.FSVM_Classifier(C = C, gamma = gamma, nu = nu, membership = 'SVDD')
    clf.fit(X_train, y_train)

    scorerMCC = metrics.make_scorer(metrics.matthews_corrcoef)
    scorerSP = metrics.make_scorer(specificity_score)
    scorerPR = metrics.make_scorer(metrics.precision_score)
    scorerSE = metrics.make_scorer(metrics.recall_score)
    scorer = {'ACC':'accuracy', 'recall':scorerSE, 'roc_auc': 'roc_auc', 'MCC':scorerMCC, 'SP':scorerSP}

    five_fold = cross_validate(clf, X_train, y_train, cv = cv, scoring = scorer)

    mean_ACC = np.mean(five_fold['test_ACC'])
    mean_sensitivity = np.mean(five_fold['test_recall'])
    mean_AUC = np.mean(five_fold['test_roc_auc'])
    mean_MCC = np.mean(five_fold['test_MCC'])
    mean_SP = np.mean(five_fold['test_SP'])


    y_pred = clf.predict(X_test)
    ACC = metrics.accuracy_score(y_test, y_pred)
    sensitivity = metrics.recall_score(y_test, y_pred)
    specificity = specificity_score(y_test, y_pred)
    AUC = metrics.roc_auc_score(y_test, clf.decision_function(X_test))
    MCC = metrics.matthews_corrcoef(y_test, y_pred)

    if best_mean_ACC <= mean_ACC:
        best_mean_ACC = mean_ACC
        best_mean_sensitivity = mean_sensitivity
        best_mean_AUC = mean_AUC
        best_mean_MCC = mean_MCC
        best_mean_SP = mean_SP
        best_ACC = ACC
        best_sensitivity = sensitivity
        best_specificity = specificity
        best_AUC = AUC
        best_MCC = MCC
        best_C = C
        best_gamma = gamma
        best_nu = nu


print('five fold:')
print(best_mean_sensitivity)
print(best_mean_SP)
print(best_mean_ACC)
print(best_mean_MCC)
print(best_mean_AUC)

print('Testing set:')
print(best_sensitivity)
print(best_specificity)
print(best_ACC)
print(best_MCC)
print(best_AUC)

print('C = ', best_C)
print('g = ', best_gamma)
print('nu =', best_nu)