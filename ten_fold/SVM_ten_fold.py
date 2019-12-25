import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection
from imblearn.metrics import specificity_score

def getlabel(data):
    m = np.shape(data)[0]
    temp = np.zeros(m)
    for x in range(m):
        temp[x] = data[x][0]
    return temp

def get_label_use_thres(y, thres):
    n = np.shape(y)[0]
    temp = np.zeros(n)
    for i in range(n):
        if y[i][1] > thres:
            temp[i] = 1
        else:
            temp[i] = 0
    return temp

def get_proba(y):
    n = np.shape(y)[0]
    temp = np.zeros(n)
    for i in range(n):
        temp[i] = y[i][1]
    return temp

kernel_name = ['cosine', 'tanimoto']
for threshold in np.arange(0.1, 1, 0.05):
    print('t =', round(threshold, 3))
    for it in range(1,2):
        s = kernel_name[it]
        print(s)
        sum_ACC = 0
        sum_PR = 0
        sum_SE = 0
        sum_SP = 0
        sum_MCC = 0
        sum_AUC = 0
        for i in range(10):
            f1 = np.loadtxt('D:/study/Bioinformatics/QSP/200p_200n/10_fold/combine_matrix/combine_' + s + '_train_' + str(i) + '.csv', delimiter = ',')
            f3 = np.loadtxt('D:/study/Bioinformatics/QSP/200p_200n/10_fold/combine_matrix/combine_' + s + '_test_' + str(i) + '.csv', delimiter = ',')
            f_test = np.loadtxt('D:/study/Bioinformatics/QSP/200p_200n/10_fold/188-bit/test/test_188-bit_' + str(i) + '.csv', delimiter = ',')
            f_train = np.loadtxt('D:/study/Bioinformatics/QSP/200p_200n/10_fold/188-bit/train/train_188-bit_' + str(i) + '.csv', delimiter = ',')

            test_label = getlabel(f_test)
            train_label = getlabel(f_train)        
            np.set_printoptions(suppress = True)
            gram = f1
            y_train = train_label
            gram_test = f3
            y_test = test_label

            clf = svm.SVC(kernel = 'precomputed', probability = True)
            clf.fit(gram, y_train)

            y_pred_proba = clf.predict_proba(gram_test)
            y_proba = get_proba(y_pred_proba)
            y_pred = get_label_use_thres(y_pred_proba, 0.5)

            ACC = metrics.accuracy_score(y_test, y_pred)
            precision = metrics.precision_score(y_test, y_pred)
            sensitivity = metrics.recall_score(y_test, y_pred)
            specificity = specificity_score(y_test, y_pred)
            AUC = metrics.roc_auc_score(y_test, y_proba)
            MCC = metrics.matthews_corrcoef(y_test, y_pred)

            sum_ACC = sum_ACC + ACC
            sum_PR = sum_PR + precision
            sum_SE = sum_SE + sensitivity
            sum_SP = sum_SP + specificity
            sum_MCC = sum_MCC + MCC
            sum_AUC = sum_AUC + AUC
        print('ACC =', round(sum_ACC/10, 5))
        #print('precision =', round(sum_PR/10, 5))
        #print('sensitivity =', round(sum_SE/10, 5))
        #print('specificity =', round(sum_SP/10, 5))
        #print('MCC =', round(sum_MCC/10, 5))
        print('AUC =', round(sum_AUC/10, 5))