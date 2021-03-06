import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection
from imblearn.metrics import specificity_score


f1 = np.loadtxt('C:/学习/Bioinformatics/AMP/kernel_matrix/KM_train_cosine/combine_cosine_train.csv', delimiter = ',')
f2 = np.loadtxt('C:/学习/Bioinformatics/AMP/train_label.csv', delimiter = ',')
f3 = np.loadtxt('C:/学习/Bioinformatics/AMP/kernel_matrix/KM_test_cosine/combine_cosine_test.csv', delimiter = ',')
f4 = np.loadtxt('C:/学习/Bioinformatics/AMP/test_label.csv', delimiter = ',')

np.set_printoptions(suppress = True)
gram = f1
y_train = f2
gram_test = f3
y_test = f4

clf = svm.SVC(kernel = 'precomputed', probability = False)
clf.fit(gram, y_train)

y_pred = clf.predict(gram_test)

ACC = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
sensitivity = metrics.recall_score(y_test, y_pred)
specificity = specificity_score(y_test, y_pred)
AUC = metrics.roc_auc_score(y_test, clf.decision_function(gram_test))
MCC = metrics.matthews_corrcoef(y_test, y_pred)

print('ACC =', round(ACC, 3))
print('precision =', round(precision, 3))
print('sensitivity =', round(sensitivity, 3))
print('specificity =', round(specificity, 3))
print('MCC =', round(MCC, 3))
print('AUC =', round(AUC, 3))