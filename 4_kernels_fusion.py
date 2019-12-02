import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection


f1 = np.loadtxt('C:/学习/Bioinformatics/QSP/144p_179n/kernel_matrix/combine_all_train.csv', delimiter = ',')
f2 = np.loadtxt('C:/学习/Bioinformatics/QSP/144p_179n/train_label.csv', delimiter = ',')
f3 = np.loadtxt('C:/学习/Bioinformatics/QSP/144p_179n/kernel_matrix/combine_all_test.csv', delimiter = ',')
f4 = np.loadtxt('C:/学习/Bioinformatics/QSP/144p_179n/test_label.csv', delimiter = ',')

np.set_printoptions(suppress = True)
gram = f1
y_train = f2
gram_test = f3
y_test = f4

clf = svm.SVC(kernel = 'precomputed', probability = True)
clf.fit(gram, y_train)
scores = model_selection.cross_val_score(clf, gram, y_train, cv = 10, scoring = 'accuracy')
ten_fold_ACC = np.mean(scores)


y_pred = clf.predict(gram_test)
y_pred_proba = clf.predict_proba(gram_test)
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
fpr, tpr, thresholds = metrics.roc_curve(y_test, clf.decision_function(gram_test))
AUC = metrics.auc(fpr, tpr)

#plt.plot(fpr, tpr, marker = 'o')
#plt.ylabel('TPR')
#plt.xlabel('FPR')
#plt.show()

print('ACC =', ACC)
print('precision =', precision)
print('sensitivity =', sensitivity)
print('specificity =', specificity)
print('F1_score =', F1_score)
#print(metrics.classification_report(y_test, y_pred))
#print('AUC =', metrics.roc_auc_score(y_test, clf.decision_function(gram_test)))
print('MCC =', metrics.matthews_corrcoef(y_test, y_pred))
#print('AUC =', AUC)
#print(y_pred_proba)
#print(y_pred)
print(scores)
print(ten_fold_ACC)