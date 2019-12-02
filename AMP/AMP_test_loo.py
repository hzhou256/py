import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection
import win32api, win32con


def get_label_use_thres(y, thres):
    n = np.shape(y)[0]
    temp = np.zeros(n)
    for i in range(n):
        if y[i][1] > thres:
            temp[i] = 1
        else:
            temp[i] = 0
    return temp

f1 = np.loadtxt('D:/Study/Bioinformatics/AMP/kernel_matrix/KM_train_cosine/combine_cosine_train.csv', delimiter = ',')
f2 = np.loadtxt('D:/Study/Bioinformatics/AMP/train_label.csv', delimiter = ',')

np.set_printoptions(suppress = True)
gram = f1
y_train = f2

clf = svm.SVC(kernel = 'precomputed', probability = True)
clf.fit(gram, y_train)

loo = model_selection.LeaveOneOut()
y_cv = model_selection.cross_val_predict(clf, gram, y_train, cv = loo, method = 'predict_proba', n_jobs = -1)

max_MCC = 0
max_ACC = 0
max_SP = 0
max_SN = 0
best_t = 0

for threshold in np.arange(0.1, 1, 0.001):
    y_pred = get_label_use_thres(y_cv, threshold)

    confusion_matrix = metrics.confusion_matrix(y_train, y_pred)
    TP = confusion_matrix[0][0]
    FP = confusion_matrix[0][1]
    FN = confusion_matrix[1][0]
    TN = confusion_matrix[1][1]

    ACC = (TP + TN) / (TP + TN + FP + FN)
    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    MCC = metrics.matthews_corrcoef(y_train, y_pred)

    if max_MCC < MCC:
        max_MCC = MCC
        max_ACC = ACC
        max_SP = SP
        max_SN = SN
        best_t = threshold

print('best_ACC =', max_ACC)
print('best_SP =', max_SP)
print('best_SN =', max_SN)
print('best_MCC =', max_MCC)
print('best_threshold =', best_t)

win32api.MessageBox(0, "运行完毕！", "提醒", win32con.MB_OK)