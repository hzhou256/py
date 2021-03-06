import csv
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn import model_selection
from scipy.spatial.distance import cdist
from imblearn.metrics import specificity_score
from hsic_kernel_weights_norm import hsic_kernel_weights_norm


G_list = [0.25, 4, 0.5, 0.03125, 0.125]
methods_name = ['188-bit', 'AAC', 'ASDC', 'CKSAAP', 'DPC']
for it in range(0, 5):
    name = methods_name[it]
    print(name)
    X_train = np.loadtxt('D:/Study/Bioinformatics/补实验/AMP/features/train_'+name+'.csv', delimiter = ',')
    X_test = np.loadtxt('D:/Study/Bioinformatics/补实验/AMP/features/710/710_'+name+'.csv', delimiter = ',')

    scaler = preprocessing.MinMaxScaler(feature_range = (0, 1)).fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = np.loadtxt('D:/Study/Bioinformatics/补实验/AMP/features/train_label.csv', delimiter = ',')
    y_test = np.loadtxt('D:/Study/Bioinformatics/补实验/AMP/features/710/label_710.csv', delimiter = ',')

    n_samples = len(y_train)

    K_test = metrics.pairwise.rbf_kernel(X_test, X_train, gamma=G_list[it])
    print(K_test)

    with open('D:/Study/Bioinformatics/补实验/AMP/kernels/710/K_710_'+name+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in K_test:
            writer.writerow(row)
        csvfile.close()








