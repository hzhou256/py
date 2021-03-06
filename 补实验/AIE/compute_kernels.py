import csv
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn import model_selection
from scipy.spatial.distance import cdist
from imblearn.metrics import specificity_score
from hsic_kernel_weights_norm import hsic_kernel_weights_norm


def tanimoto_base(p_vec, q_vec):
    pq = np.dot(p_vec, q_vec)
    p_square = np.square(np.linalg.norm(p_vec))
    q_square = np.square(np.linalg.norm(q_vec))
    d = pq / (p_square + q_square - pq)
    return d

def tanimoto(X, Y):
    K = cdist(X, Y, tanimoto_base)
    return K

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


methods_name = ['AAC', 'ASDC', 'CKSAAP', 'DPC', '188-bit']
for it in range(4, 5):
    name = methods_name[it]
    print(name)
    X_train = np.loadtxt('D:/Study/Bioinformatics/补实验/AIE/features/train_'+name+'.csv', delimiter = ',')
    X_test = np.loadtxt('D:/Study/Bioinformatics/补实验/AIE/features/test_'+name+'.csv', delimiter = ',')
    


    y_train = np.loadtxt('D:/Study/Bioinformatics/补实验/AIE/features/train_label.csv', delimiter = ',')
    y_test = np.loadtxt('D:/Study/Bioinformatics/补实验/AIE/features/test_label.csv', delimiter = ',')

    n_samples = len(y_train)

    K_train = tanimoto(X_train, X_train)
    K_test = tanimoto(X_test, X_train)
    with open('D:/Study/Bioinformatics/补实验/AIE/kernels/K_train_'+name+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in K_train:
            writer.writerow(row)
        csvfile.close()
    with open('D:/Study/Bioinformatics/补实验/AIE/kernels/K_test_'+name+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in K_test:
            writer.writerow(row)
        csvfile.close()


    #kernel_list = [K_train_ASDC, K_train_CKSAAP]
    #weight_v = hsic_kernel_weights_norm(kernel_list, y_train, 1, 0.01, 0.001)






