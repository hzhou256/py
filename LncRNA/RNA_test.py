import numpy as np
import Fuzzy_SVM, membership
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold
from sklearn import metrics, preprocessing, svm
from imblearn.metrics import specificity_score


def get_feature(file):
    m = np.shape(file)[0]
    n = np.shape(file)[1]
    data = np.zeros((m, n-1))
    for index in range(m):
        data[index] = file[index][1:]
    return data


f1 = np.loadtxt('E:/Study/Bioinformatics/RNA/dataset/train.csv', delimiter = ',', skiprows = 1)
X_train = get_feature(f1)
y_train = f1[:, 0]

f2 = np.loadtxt('E:/Study/Bioinformatics/RNA/dataset/test.csv', delimiter = ',', skiprows = 1)
X_test = get_feature(f2)
y_test = f2[:, 0]


nu = 0.1
cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
parameters = {'C': np.logspace(-10, 10, base = 2, num = 21), 'gamma': np.logspace(5, -15, base = 2, num = 21)}


#grid = GridSearchCV(Fuzzy_SVM.FSVM_Classifier(membership = 'None'), parameters, n_jobs = -1, cv = cv, verbose = 1)
grid = GridSearchCV(Fuzzy_SVM.FSVM_Classifier(membership = 'SVDD', nu = nu), parameters, n_jobs = -1, cv = cv, verbose = 1)
grid.fit(X_train, y_train)
gamma = grid.best_params_['gamma']
C = grid.best_params_['C']


#clf = Fuzzy_SVM.FSVM_Classifier(C = C, gamma = gamma, membership = 'None')
clf = Fuzzy_SVM.FSVM_Classifier(C = C, gamma = gamma, membership = 'SVDD', nu = nu)
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

print('five fold:')
print(mean_sensitivity)
print(mean_SP)
print(mean_ACC)
print(mean_MCC)
print(mean_AUC)

y_pred = clf.predict(X_test)
ACC = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
sensitivity = metrics.recall_score(y_test, y_pred)
specificity = specificity_score(y_test, y_pred)
AUC = metrics.roc_auc_score(y_test, clf.decision_function(X_test))
MCC = metrics.matthews_corrcoef(y_test, y_pred)

print('Testing set:')
print(sensitivity)
print(specificity)
print(ACC)
print(MCC)
print(AUC)

print('C = ', C)
print('g = ', gamma)
