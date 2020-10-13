import csv
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn import model_selection
from scipy.spatial.distance import cdist
from imblearn.metrics import specificity_score
from hsic_kernel_weights_norm import hsic_kernel_weights_norm




G_list = []
methods_name = ['AAC', 'ASDC', 'CKSAAP', 'DPC', '188-bit']
for it in range(0, 5):
    name = methods_name[it]
    print(name)
    X_train = np.loadtxt('D:/Study/Bioinformatics/补实验/AVP/features/train_'+name+'.csv', delimiter = ',')
    X_test = np.loadtxt('D:/Study/Bioinformatics/补实验/AVP/features/test_'+name+'.csv', delimiter = ',')


    y_train = np.loadtxt('D:/Study/Bioinformatics/补实验/AVP/features/train_label.csv', delimiter = ',')
    y_test = np.loadtxt('D:/Study/Bioinformatics/补实验/AVP/features/test_label.csv', delimiter = ',')

    K_train = metrics.pairwise.rbf_kernel(X_train, X_train, gamma=G_list[it])
    K_test = metrics.pairwise.rbf_kernel(X_test, X_train, gamma=G_list[it])
    
    with open('D:/Study/Bioinformatics/补实验/AVP/kernels/K_train_'+name+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in K_train:
            writer.writerow(row)
        csvfile.close()
    with open('D:/Study/Bioinformatics/补实验/AVP/kernels/K_test_'+name+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in K_test:
            writer.writerow(row)
        csvfile.close()








