import numpy as np
from sklearn import preprocessing

from mRNA import Class_KDVM_multi

dataset = ['CKSNAP', 'DNC', 'Kmer4', 'Kmer1234', 'NAC', 'RCKmer', 'TNC']
for i in range(4, 5):
    name = dataset[i]
    print(name)
    f1 = np.loadtxt('D:/Study/Bioinformatics/王浩/data and code/data/feature/mRNA/mRNA_' + name + '.csv', delimiter=',')
    f2 = np.loadtxt('D:/Study/Bioinformatics/王浩/data and code/data/feature/mRNA/mRNA_label.csv', delimiter=',',
                    skiprows=1)
    X = f1
    y = f2

    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(X)
    X = scaler.transform(X)

    clf = Class_KDVM_multi.KDVM(kernel='rbf', gamma=1, n_neighbors=20)
    clf.fit(X, y)

    # y_dec = clf.decision_function(X)
    # print(y_dec)
