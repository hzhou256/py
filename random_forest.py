import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


f1 = np.loadtxt("C:/学习/Bioinformatics/QSP/300p_300n/CTD/train_CTD.csv", delimiter = ',', skiprows = 1)
f2 = np.loadtxt('C:/学习/Bioinformatics/QSP/300p_300n/train_label.csv', delimiter = ',')
f3 = np.loadtxt("C:/学习/Bioinformatics/QSP/300p_300n/CTD/test_CTD.csv", delimiter = ',', skiprows = 1)
f4 = np.loadtxt('C:/学习/Bioinformatics/QSP/300p_300n/test_label.csv', delimiter = ',')

def get_matrix(file):
    m = np.shape(file)[0]
    n = np.shape(file)[1]
    d = np.zeros((m, n-1))
    for index in range(m):
        d[index] = file[index][1:]
    return d

X_train = get_matrix(f1)
X_test = get_matrix(f3)
y_train = f2
y_test = f4

RF = RandomForestClassifier(n_estimators = 30, criterion = 'gini', min_samples_leaf = 1)
RF.fit(X_train, y_train)
y_pred = RF.predict(X_test)

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

TP = confusion_matrix[0][0]
FP = confusion_matrix[0][1]
FN = confusion_matrix[1][0]
TN = confusion_matrix[1][1]

ACC = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
F1_score = 2 * precision * sensitivity / (precision + sensitivity)
AUC = metrics.roc_auc_score(y_test, y_pred)


print('ACC =', ACC)
print('precision =', precision)
print('sensitivity =', sensitivity)
print('specificity =', specificity)
print('F1_score =', F1_score)
print('MCC =', metrics.matthews_corrcoef(y_test, y_pred))
print('AUC =', AUC)
