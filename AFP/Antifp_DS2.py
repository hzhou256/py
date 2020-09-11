import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection
from imblearn.metrics import specificity_score


def get_matrix(file):
    m = np.shape(file)[0]
    n = np.shape(file)[1]
    d = np.zeros((m, n-1))
    for index in range(m):
        d[index] = file[index][1:]
    return d

def get_y_score(y_proba):
    n = np.shape(y_proba)[0]
    temp = np.zeros(n)
    for i in range(n):
        temp[i] = y_proba[i][1]
    return temp

def get_AUPR(y_true, y_score):
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_score, pos_label = 1)
    AUPR = metrics.auc(recall, precision)
    return AUPR

dataset_name = ['Antifp_Main', 'Antifp_DS1', 'Antifp_DS2']
for ds in range(2,3):
    name_ds = dataset_name[ds]
    print(name_ds)



    f1 = np.loadtxt('/kernel_matrix/' + name_ds + '/KM_train_tanimoto/combine_tanimoto_HSIC_train.csv', delimiter = ',')
    f2 = np.loadtxt('/kernel_matrix/' + name_ds + '/train_label.csv', delimiter = ',')
    f3 = np.loadtxt('/kernel_matrix/' + name_ds + '/KM_test_tanimoto/combine_tanimoto_HSIC_test.csv', delimiter = ',')
    f4 = np.loadtxt('/kernel_matrix/' + name_ds + '/test_label.csv', delimiter = ',')

    gram = f1
    y_train = f2
    gram_test = f3
    y_test = f4

    clf = svm.SVC(C = 3.237, kernel = 'precomputed', probability = True)
    clf.fit(gram, y_train)

    y_score = clf.predict_proba(gram_test)
    y_score = get_y_score(y_score)
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_score)

    y_pred = clf.predict(gram_test)
    ACC = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    sensitivity = metrics.recall_score(y_test, y_pred)
    specificity = specificity_score(y_test, y_pred)
    AUC = metrics.roc_auc_score(y_test, y_score)
    MCC = metrics.matthews_corrcoef(y_test, y_pred)
    AUPR = get_AUPR(y_test, y_score)


    print('SN =', sensitivity)
    print('SP =', specificity)
    print('ACC =', ACC)
    print('MCC =', MCC)
    print('AUC =', AUC)
    print('AUPR =', AUPR)

