import collections
import numpy as np
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold


dataset = ['australian', 'breastw', 'diabetes', 'german', 'heart', 'ionosphere', 'sonar', 'mushroom', 'bupa', 'transfusion', 'spam']
for i in range(0, 1):
    name = dataset[i]
    print(name)
    f1 = np.loadtxt('E:/Study/Bioinformatics/UCI/' + name + '/data.csv', delimiter = ',')
    X = f1[:, 0:-1]
    y = f1[:, -1]

    target = np.array([0, 1, 2, 3])
    index = np.arange(len(y))
    a = X[target[:]]
    b = np.zeros(np.shape(a))
    b[0:4] = a
    print(b)