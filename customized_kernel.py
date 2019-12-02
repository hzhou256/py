import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection
from imblearn.metrics import specificity_score


f1 = np.loadtxt('C:/学习/Bioinformatics/QSP/200p_200n/kernel_matrix/KM_train_tanimoto/combine_tanimoto_train.csv', delimiter = ',')
f2 = np.loadtxt('C:/学习/Bioinformatics/QSP/200p_200n/train_label.csv', delimiter = ',')
f3 = np.loadtxt('C:/学习/Bioinformatics/QSP/200p_200n/kernel_matrix/KM_test_tanimoto/combine_tanimoto_test.csv', delimiter = ',')
f4 = np.loadtxt('C:/学习/Bioinformatics/QSP/200p_200n/test_label.csv', delimiter = ',')

np.set_printoptions(suppress = True)
gram = f1
y_train = f2
gram_test = f3
y_test = f4

clf = svm.SVC(kernel = 'precomputed', probability = True)
clf.fit(gram, y_train)

y_pred = clf.predict(gram_test)
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

print('ACC =', round(ACC, 3))
#print('precision =', round(precision, 3))
print('sensitivity =', round(sensitivity, 3))
print('specificity =', round(specificity, 3))
#print('F1_score =', round(F1_score, 3))
print('MCC =', round(metrics.matthews_corrcoef(y_test, y_pred), 3))
print('AUC =', round(AUC, 3))

scoreACC = metrics.make_scorer(metrics.accuracy_score)
scorerMCC = metrics.make_scorer(metrics.matthews_corrcoef)
scorerSP = metrics.make_scorer(specificity_score)
scorerPR = metrics.make_scorer(metrics.precision_score)
scorerSE = metrics.make_scorer(metrics.recall_score)
scorerAUC = metrics.make_scorer(metrics.roc_auc_score)


cv = model_selection.StratifiedKFold(n_splits = 10, shuffle = True)


scorer = {'ACC':scoreACC, 'precision': scorerPR, 'recall':scorerSE, 'roc_auc': 'roc_auc', 'MCC':scorerMCC, 'SP':scorerSP}
ten_fold = model_selection.cross_validate(clf, gram, y_train, cv = cv, scoring = scorer)

mean_ACC = np.mean(ten_fold['test_ACC'])
mean_precision = np.mean(ten_fold['test_precision'])
mean_sensitivity = np.mean(ten_fold['test_recall'])
mean_AUC = np.mean(ten_fold['test_roc_auc'])
mean_MCC = np.mean(ten_fold['test_MCC'])
mean_SP = np.mean(ten_fold['test_SP'])

print('\n')
print('10-fold:')
print('ACC =', round(mean_ACC, 3))
#print('precision =', round(mean_precision, 3))
print('sensitivity =', round(mean_sensitivity, 3))
print('specificity =', round(mean_SP, 3))
print('MCC = ', round(mean_MCC, 3))
print('AUC = ', round(mean_AUC, 3))