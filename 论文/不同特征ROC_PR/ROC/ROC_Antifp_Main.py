import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection
import matplotlib.pyplot as plt 

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

dataset_name = ['Antifp_Main', 'Antifp_DS1', 'Antifp_DS2']
methods_name = ['188-bit', 'AAC', 'ASDC', 'CKSAAP', 'DPC']
c_list = [99.760, 37.302, 8.286, 2.023, 2.649]
auc_list = [0.8964, 0.9236, 0.9532, 0.9650, 0.9645]

for ds in range(1):
    name_ds = dataset_name[ds]
    print(name_ds)
    for it in range(5):
        name = methods_name[it]

        f1 = np.loadtxt('E:/Study/Bioinformatics/AFP/kernel_matrix/' + name_ds + '/KM_train_tanimoto/KM_tanimoto_' + name + '_train.csv', delimiter = ',')
        f2 = np.loadtxt('E:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds + '/train_label.csv', delimiter = ',')
        f3 = np.loadtxt('E:/Study/Bioinformatics/AFP/kernel_matrix/' + name_ds + '/KM_test_tanimoto/KM_tanimoto_' + name + '_test.csv', delimiter = ',')
        f4 = np.loadtxt('E:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds + '/test_label.csv', delimiter = ',')

        np.set_printoptions(suppress = True)
        gram = f1
        y_train = f2
        gram_test = f3
        y_test = f4

        clf = svm.SVC(C = c_list[it], kernel = 'precomputed', probability = True)
        clf.fit(gram, y_train)

        y_score = clf.predict_proba(gram_test)
        y_score = get_y_score(y_score)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
        plt.plot(fpr, tpr, label = name + ' - AUC: ' + str(auc_list[it]))  

    plt.legend()
    plt.title(name_ds)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.savefig("E:/论文/图表/ROC_PR/不同特征/ROC_" + name_ds + ".png")
    plt.show()