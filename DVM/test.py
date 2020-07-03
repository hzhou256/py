import collections
import numpy as np
import Class_KDVM_fast
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold


f1 = np.loadtxt('E:/Study/Bioinformatics/UCI/australian/data.csv', delimiter = ',')
X = f1[:, 0:-1]
y = f1[:, -1]

for i in range(len(y)):
    if y[i] == -1:
        y[i] = 0

cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
parameters = {'gamma': np.logspace(5, -15, base = 2, num = 21)}

grid = GridSearchCV(Class_KDVM_fast.KDVM(kernel = 'rbf'), parameters, n_jobs = -1, cv = cv, verbose = 1)
grid.fit(X, y)
gamma = grid.best_params_['gamma']

clf = Class_KDVM_fast.KDVM(kernel = 'rbf', gamma = gamma)
five_fold = cross_validate(clf, X, y, cv = cv, scoring = 'accuracy', n_jobs = -1)
mean_ACC = np.mean(five_fold['test_score'])
print(mean_ACC)

'''
clf = Class_KDVM_fast.KDVM()
y_pred = clf.fit_predict(X, y, X)
ACC = metrics.accuracy_score(y, y_pred)
print(ACC)
'''
