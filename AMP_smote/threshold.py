import numpy as np
from sklearn import metrics
from sklearn import model_selection


def get_label_use_thres(y, thres):
    n = np.shape(y)[0]
    temp = np.zeros(n)
    for i in range(n):
        if y[i] > thres:
            temp[i] = 1
        else:
            temp[i] = 0
    return temp

def get_y(y_proba, c):
    n = np.shape(y_proba)[0]
    temp = np.zeros(n)
    for i in range(n):
        temp[i] = y_proba[i][c]
    return temp

f1 = np.loadtxt('D:/Study/Bioinformatics/AMP_smote/multiple_label_1.csv', delimiter = ',')
f2 = np.loadtxt('D:/Study/Bioinformatics/AMP_smote/kernel_matrix_1/KM_train_cosine/y_proba.csv', delimiter = ',')

y_train = f1
y_proba = f2

np.set_printoptions(suppress = True)

for c in range(5):
    best_MCC = 0
    best_ACC = 0
    best_SP = 0
    best_SN = 0
    best_PR = 0
    best_t = 0
    y_prob = get_y(y_proba, c)
    for threshold in np.arange(0.1, 1, 0.001):
        y_pred = get_label_use_thres(y_prob, threshold)
        y_true = get_y(y_train, c)

        confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
        TP = confusion_matrix[0][0]
        FP = confusion_matrix[0][1]
        FN = confusion_matrix[1][0]
        TN = confusion_matrix[1][1]

        ACC = (TP + TN) / (TP + TN + FP + FN)
        SN = TP / (TP + FN)
        SP = TN / (TN + FP)
        PR = TP / (TP + FP)
        MCC = metrics.matthews_corrcoef(y_true, y_pred)
        if best_MCC < MCC:
            best_MCC = MCC
            best_ACC = ACC
            best_SP = SP
            best_SN = SN
            best_PR = PR
            best_t = threshold
    print('class_' + str(c+1) + ':')
    print('ACC =', best_ACC)
    print('SN =', best_SN)
    print('SP =', best_SP)
    print('Precision =', best_PR)
    print('MCC =', best_MCC)
    print('threshold =', best_t)