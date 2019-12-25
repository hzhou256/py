import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection
from imblearn.metrics import specificity_score


f1 = np.loadtxt('D:/Study/Bioinformatics/QSP_new/kernel_matrix_3/KM_train_tanimoto/combine_tanimoto_train.csv', delimiter = ',')
f2 = np.loadtxt('D:/Study/Bioinformatics/QSP_new/train_label.csv', delimiter = ',')

np.set_printoptions(suppress = True)
gram = f1
y_train = f2

clf = svm.SVC(C = 4.094915062380419, kernel = 'precomputed', probability = True)
clf.fit(gram, y_train)

scorerMCC = metrics.make_scorer(metrics.matthews_corrcoef)
scorerSP = metrics.make_scorer(specificity_score)
scorerPR = metrics.make_scorer(metrics.precision_score)
scorerSE = metrics.make_scorer(metrics.recall_score)

cv = model_selection.StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)

scorer = {'ACC':'accuracy', 'precision': scorerPR, 'recall':scorerSE, 'roc_auc': 'roc_auc', 'MCC':scorerMCC, 'SP':scorerSP}
five_fold = model_selection.cross_validate(clf, gram, y_train, cv = cv, scoring = scorer)

mean_ACC = np.mean(five_fold['test_ACC'])
mean_precision = np.mean(five_fold['test_precision'])
mean_sensitivity = np.mean(five_fold['test_recall'])
mean_AUC = np.mean(five_fold['test_roc_auc'])
mean_MCC = np.mean(five_fold['test_MCC'])
mean_SP = np.mean(five_fold['test_SP'])

#print('precision =', round(mean_precision, 3))
print('SN =', mean_sensitivity)
print('SP =', mean_SP)
print('ACC =', mean_ACC)
print('MCC = ', mean_MCC)
print('AUC = ', mean_AUC)