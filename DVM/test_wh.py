import collections
import numpy as np
import Class_KDVM_Lap
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold


f1 = np.loadtxt('D:/Study/Bioinformatics/wh/DPC/train_DPC.csv', delimiter = ',')
f2 = np.loadtxt('D:/Study/Bioinformatics/wh/DPC/test_DPC.csv', delimiter = ',')
X_train = f1[:, 0:-1]
y_train = f1[:, -1]
X_test = f2[:, 0:-1]
y_test = f2[:, -1]

#scaler = preprocessing.MinMaxScaler(feature_range = (-1, 1)).fit(X)
#X_train = scaler.transform(X) # 特征标准化

cnt = dict(collections.Counter(y_train))
n_class = len(cnt)
max_val = int(min(cnt.values())/5*4)
num = int((max_val - max_val%5)/5)


cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
parameters = {'gamma': np.logspace(5, -15, base = 2, num = 21), 'n_neighbors': np.linspace(5, max_val, num = num, dtype = int)}
#parameters = {'gamma': np.logspace(5, -15, base = 2, num = 21), 'n_neighbors': np.linspace(10, max_val, num = int(num/2), dtype = int)}


grid = GridSearchCV(Class_KDVM_Lap.KDVM(kernel = 'rbf'), parameters, n_jobs = -1, cv = cv, verbose = 2)
grid.fit(X_train, y_train)
gamma = grid.best_params_['gamma']
n_neighbors = grid.best_params_['n_neighbors']

clf = Class_KDVM_Lap.KDVM(kernel = 'rbf', gamma = gamma, n_neighbors = n_neighbors)
five_fold = cross_validate(clf, X_train, y_train, cv = cv, scoring = 'accuracy', n_jobs = -1)
mean_ACC = np.mean(five_fold['test_score'])
print(mean_ACC)

print(gamma)
print(n_neighbors)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
ACC = metrics.accuracy_score(y_test, y_pred)
print(ACC)


'''
gamma = 0.0009765625
n_neighbors = 17
cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
clf = Class_KDVM_Lap.KDVM(kernel = 'rbf', gamma = gamma, n_neighbors = n_neighbors)
five_fold = cross_validate(clf, X, y, cv = cv, scoring = 'accuracy', n_jobs = -1)
mean_ACC = np.mean(five_fold['test_score'])
print(mean_ACC)
'''
'''
cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)    
parameters_1 = {'n_neighbors': np.linspace(5, max_val, num = num, dtype = int)}
#parameters_1 = {'n_neighbors': np.linspace(10, 100, num = 10, dtype = int)}

grid_1 = GridSearchCV(Class_KDVM_Lap.KDVM(kernel = 'linear'), parameters_1, n_jobs = -1, cv = cv, verbose = 2)
grid_1.fit(X, y)
n_neighbors = grid_1.best_params_['n_neighbors']

clf = Class_KDVM_Lap.KDVM(kernel = 'linear', n_neighbors = n_neighbors)
five_fold = cross_validate(clf, X, y, cv = cv, scoring = 'accuracy', n_jobs = -1)
mean_ACC = np.mean(five_fold['test_score'])
print(mean_ACC)

print(n_neighbors)
'''