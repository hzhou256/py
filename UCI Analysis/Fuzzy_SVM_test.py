import sys
path='D:/Program Files/libsvm_weights-3.23/python'
sys.path.append(path)
import numpy as np
import membership, My_Fuzzy_SVM
from sklearn import metrics
from imblearn.metrics import specificity_score
from sklearn.model_selection import GridSearchCV, cross_validate


def get_feature(file):
    m = np.shape(file)[0]
    n = np.shape(file)[1]
    data = np.zeros((m, n-1))
    for index in range(m):
        data[index] = file[index][1:]
    return data


dataset = ['ionosphere', 'german', 'sonar']
name = dataset[2]

f1 = np.loadtxt('E:/Study/Bioinformatics/UCI/' + name + '/X_train.csv', delimiter = ',', skiprows = 1)
X_train = get_feature(f1)
y_train = f1[:, 0]

f2 = np.loadtxt('E:/Study/Bioinformatics/UCI/' + name + '/X_test.csv', delimiter = ',', skiprows = 1)
X_test = get_feature(f2)
y_test = f2[:, 0]

#parameters = {'C': np.logspace(-10, 10, base = 2, num = 21), 'gamma': np.logspace(5, -15, base = 2, num = 21), 'nu': np.linspace(0, 1, num = 10)}
#parameters = {'C': np.logspace(-10, 10, base = 2, num = 21), 'gamma': np.logspace(5, -15, base = 2, num = 21), 'nu': [0.5]}
parameters = {'C': np.logspace(-10, 10, base = 2, num = 21), 'gamma': np.logspace(5, -15, base = 2, num = 21)}
#grid = GridSearchCV(My_Fuzzy_SVM.FSVM_Classifier(membership = 'SVDD'), parameters, n_jobs = -1, cv = 5, verbose = 1)
grid = GridSearchCV(My_Fuzzy_SVM.FSVM_Classifier(membership = 'None'), parameters, n_jobs = -1, cv = 5, verbose = 1)
grid.fit(X_train, y_train)
gamma = grid.best_params_['gamma']
C = grid.best_params_['C']
#nu = grid.best_params_['nu']
C =  16.0
gamma =  0.0001220703125
clf = My_Fuzzy_SVM.FSVM_Classifier(C = C, gamma = gamma, nu = nu, membership = 'SVDD')
#clf = My_Fuzzy_SVM.FSVM_Classifier(C = C, gamma = gamma, membership = 'None')
clf.fit(X_train, y_train)

scorerMCC = metrics.make_scorer(metrics.matthews_corrcoef)
scorerSP = metrics.make_scorer(specificity_score)
scorerPR = metrics.make_scorer(metrics.precision_score)
scorerSE = metrics.make_scorer(metrics.recall_score)
scorer = {'ACC':'accuracy', 'recall':scorerSE, 'roc_auc': 'roc_auc', 'MCC':scorerMCC, 'SP':scorerSP}
five_fold = cross_validate(clf, X_train, y_train, cv = 5, scoring = scorer)
mean_ACC = np.mean(five_fold['test_ACC'])
mean_sensitivity = np.mean(five_fold['test_recall'])
mean_AUC = np.mean(five_fold['test_roc_auc'])
mean_MCC = np.mean(five_fold['test_MCC'])
mean_SP = np.mean(five_fold['test_SP'])

print('five fold:')
print('SN =', mean_sensitivity)
print('SP =', mean_SP)
print('ACC =', mean_ACC)
print('MCC = ', mean_MCC)
print('AUC = ', mean_AUC)

y_pred = clf.predict(X_test)
ACC = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
sensitivity = metrics.recall_score(y_test, y_pred)
specificity = specificity_score(y_test, y_pred)
AUC = metrics.roc_auc_score(y_test, clf.decision_function(X_test))
MCC = metrics.matthews_corrcoef(y_test, y_pred)

print('Testing set:')
print('SN =', sensitivity)
print('SP =', specificity)
print('ACC =', ACC)
print('MCC =', MCC)
print('AUC =', AUC)


print('C = ', C)
print('g = ', gamma)
#print('nu =', nu)
