import collections
import numpy as np
import Class_DVM
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors



f1 = np.loadtxt('E:/Study/Bioinformatics/UCI/australian/data.csv', delimiter = ',')
X = f1[:, 0:-1]
y = f1[:, -1]

for i in range(len(y)):
    if y[i] == -1:
        y[i] = 0

clf = Class_DVM.DVM()
X_test = X
y_pred = clf.predict(X, y, X_test)
print(y_pred)
ACC = metrics.accuracy_score(y, y_pred)
print(ACC)