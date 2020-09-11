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


methods_name = ['AAC', 'ASDC', 'CKSAAP', 'DPC', '188-bit']
for it in range(0, 5):
    name = methods_name[it]
    print(name)
    X_train = np.loadtxt('D:/Study/Bioinformatics/补实验/AVP/features/train_'+name+'.csv', delimiter = ',')
    X_test = np.loadtxt('D:/Study/Bioinformatics/补实验/AVP/features/novel/novel_'+name+'.csv', delimiter = ',')


    y_train = np.loadtxt('D:/Study/Bioinformatics/补实验/AVP/features/train_label.csv', delimiter = ',')
    y_test = np.loadtxt('D:/Study/Bioinformatics/补实验/AVP/features/novel/novel_label.csv', delimiter = ',')


    K_test = tanimoto(X_test, X_train)

    with open('D:/Study/Bioinformatics/补实验/AVP/kernels/novel/K_novel_'+name+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in K_test:
            writer.writerow(row)
        csvfile.close()








