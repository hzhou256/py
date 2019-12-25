import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection
from imblearn.metrics import specificity_score


def get_y(y_proba):
    n = np.shape(y_proba)[0]
    temp = np.zeros(n)
    for i in range(n):
        temp[i] = y_proba[i][1]
    return temp

def get_label_use_thres(y, thres):
    n = np.shape(y)[0]
    temp = np.zeros(n)
    for i in range(n):
        if y[i] > thres:
            temp[i] = 1
        else:
            temp[i] = 0
    return temp

f1 = np.loadtxt('D:/Study/Bioinformatics/AFP/kernel_matrix_selected/Antifp_Main/KM_train_tanimoto/combine_tanimoto_train.csv', delimiter = ',')
f2 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix_selected/Antifp_Main/train_label.csv', delimiter = ',')
f3 = np.loadtxt('D:/Study/Bioinformatics/AFP/kernel_matrix_selected/Antifp_Main/KM_test_tanimoto/combine_tanimoto_test.csv', delimiter = ',')
f4 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix_selected/Antifp_Main/test_label.csv', delimiter = ',')

np.set_printoptions(suppress = True)
gram = f1
y_train = f2
gram_test = f3
y_test = f4

clf = svm.SVC(kernel = 'precomputed', probability = True)
clf.fit(gram, y_train)

y_p = clf.predict_proba(gram_test)
y_proba = get_y(y_p)

max_MCC = 0
max_ACC = 0
max_SP = 0
max_SN = 0
max_AUC = 0
max_t = 0


thresholds = np.sort(y_proba)
print(thresholds)

for threshold in thresholds:
    print('thres =', threshold)
    y_pred = get_label_use_thres(y_proba, threshold)

    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    TP = confusion_matrix[0][0]
    FP = confusion_matrix[0][1]
    FN = confusion_matrix[1][0]
    TN = confusion_matrix[1][1]

    ACC = (TP + TN) / (TP + TN + FP + FN)
    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    MCC = metrics.matthews_corrcoef(y_test, y_pred)
    AUC = metrics.roc_auc_score(y_test, y_proba)

    if max_ACC < ACC:
        max_MCC = MCC
        max_ACC = ACC
        max_SP = SP
        max_SN = SN
        max_AUC = AUC
        max_t = threshold

print('ACC:')
print('best_SN =', max_SN)
print('best_SP =', max_SP)
print('best_ACC =', max_ACC)
print('best_MCC =', max_MCC)
print('best_AUC =', max_AUC)
print('best_threshold =', max_t)
