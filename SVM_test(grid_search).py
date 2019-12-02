import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection


f1 = np.loadtxt('C:/学习/Bioinformatics/QSP/99p_99n/CTD/train_CTD.csv', delimiter = ',', skiprows = 1)
f2 = np.loadtxt('C:/学习/Bioinformatics/QSP/99p_99n/train_label.csv', delimiter = ',')
f3 = np.loadtxt('C:/学习/Bioinformatics/QSP/99p_99n/CTD/test_CTD.csv', delimiter = ',', skiprows = 1)
f4 = np.loadtxt('C:/学习/Bioinformatics/QSP/99p_99n/test_label.csv', delimiter = ',')

def get_matrix(file):
    m = np.shape(file)[0]
    n = np.shape(file)[1]
    d = np.zeros((m, n-1))
    for index in range(m):
        d[index] = file[index][1:]
    return d

np.set_printoptions(suppress = True)
X_train = get_matrix(f1)
y_train = f2
X_test = get_matrix(f3)
y_test = f4

parameters = {'kernel':['rbf'],'C':np.logspace(-30, 30, base = 2), 'gamma':np.logspace(30, -30, base = 2)}
grid = model_selection.GridSearchCV(svm.SVC(), parameters, n_jobs = -1, cv = 5)
grid.fit(X_train, y_train)

cost = grid.best_params_['C']
gamma = grid.best_params_['gamma']
print('C =', cost)
print('g =', gamma)

clf = svm.SVC(C = cost, gamma = gamma, kernel = 'rbf')
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
auc = metrics.roc_auc_score(y_test, y_pred)

TP = confusion_matrix[0][0]
FP = confusion_matrix[0][1]
FN = confusion_matrix[1][0]
TN = confusion_matrix[1][1]

ACC = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
F1_score = 2 * precision * sensitivity / (precision + sensitivity)

ten_fold_ACC = model_selection.cross_val_score(clf, X_train, y_train, cv = 10, scoring = 'accuracy')
ten_fold_f1 = model_selection.cross_val_score(clf, X_train, y_train, cv = 10, scoring = 'f1')
ten_fold_precision = model_selection.cross_val_score(clf, X_train, y_train, cv = 10, scoring = 'precision')
ten_fold_sensitivity = model_selection.cross_val_score(clf, X_train, y_train, cv = 10, scoring = 'recall')
mean_ACC = np.mean(ten_fold_ACC)
mean_f1 = np.mean(ten_fold_f1)
mean_precision = np.mean(ten_fold_precision)
mean_sensitivity = np.mean(ten_fold_sensitivity)

print('\n')
print('ACC =', ACC)
print('precision =', precision)
print('sensitivity =', sensitivity)
print('specificity =', specificity)
print('F1_score =', F1_score)
print('MCC =', metrics.matthews_corrcoef(y_test, y_pred))
print('AUC =', auc)

print('\n')
print('10-fold:')
print('ACC =', mean_ACC)
print('precision =', mean_precision)
print('sensitivity =', mean_sensitivity)
print('f1 =', mean_f1)