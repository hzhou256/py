import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection
from imblearn.metrics import specificity_score


f1 = np.loadtxt('C:/学习/Bioinformatics/AMP/kernel_matrix/KM_train_tanimoto/combine_tanimoto_train.csv', delimiter = ',')
f2 = np.loadtxt('C:/学习/Bioinformatics/AMP/train_label.csv', delimiter = ',')

np.set_printoptions(suppress = True)
gram = f1
y_train = f2

clf = svm.SVC(kernel = 'precomputed', probability = False)
clf.fit(gram, y_train)

scorerMCC = metrics.make_scorer(metrics.matthews_corrcoef)
scorerSP = metrics.make_scorer(specificity_score)
scorerPR = metrics.make_scorer(metrics.precision_score)
scorerSE = metrics.make_scorer(metrics.recall_score)
scoreACC = metrics.make_scorer(metrics.accuracy_score)

cv = model_selection.StratifiedKFold(n_splits = 10, shuffle = True)

scorer = {'ACC':scoreACC, 'precision': scorerPR, 'recall':scorerSE, 'roc_auc': 'roc_auc', 'MCC':scorerMCC, 'SP':scorerSP}
ten_fold = model_selection.cross_validate(clf, gram, y_train, cv = cv, scoring = scorer)

mean_ACC = np.mean(ten_fold['test_ACC'])
mean_precision = np.mean(ten_fold['test_precision'])
mean_sensitivity = np.mean(ten_fold['test_recall'])
mean_AUC = np.mean(ten_fold['test_roc_auc'])
mean_MCC = np.mean(ten_fold['test_MCC'])
mean_SP = np.mean(ten_fold['test_SP'])

print('10-fold:')
print('ACC =', round(mean_ACC, 3))
#print('precision =', round(mean_precision, 3))
print('sensitivity =', round(mean_sensitivity, 3))
print('specificity =', round(mean_SP, 3))
print('MCC = ', round(mean_MCC, 3))
print('AUC = ', round(mean_AUC, 3))