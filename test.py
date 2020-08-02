import collections
import numpy as np
import DVM.Class_KDVM_knn
from sklearn import metrics
from sklearn import preprocessing
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold


dataset = ['wine', 'iris', 'glass', 'RNA', 'vehicle', 'abalone']
for i in range(0, 1):
    name = dataset[i]
    print(name)

    f1 = np.loadtxt('D:/Study/Bioinformatics/UCI/' + name + '/data.csv', delimiter = ',')
    X = f1[:, 0:-1]
    y = f1[:, -1]

    scaler = preprocessing.MinMaxScaler(feature_range = (-1, 1)).fit(X)
    X = scaler.transform(X) # 特征标准化

    cnt = dict(collections.Counter(y))
    n_class = len(cnt)
    max_val = int(min(cnt.values())/5*4)
    num = int((max_val - max_val%5)/5)

    
    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)

    clf = DVM.Class_KDVM_knn.KDVM(kernel = 'rbf', gamma = 1, n_neighbors = 5)
    clf.fit(X, y)
    y_prob = clf.decision_function(X)
    print(y_prob)

    

    