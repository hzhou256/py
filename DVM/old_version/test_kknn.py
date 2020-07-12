import numpy as np
import Class_KDVM_kknn
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold


dataset = ['australian', 'breastw', 'diabetes', 'german', 'heart', 'ionosphere', 'sonar', 'mushroom', 'bupa', 'blood', 'spam']
for i in range(0, 1):
    name = dataset[i]
    print(name)
    f1 = np.loadtxt('E:/Study/Bioinformatics/UCI/' + name + '/data.csv', delimiter = ',')
    X = f1[:, 0:-1]
    y = f1[:, -1]

    cnt_0, cnt_1 = 0, 0
    for i in range(len(y)):
        if y[i] == -1:
            y[i] = 0
            cnt_0 += 1
        elif y[i] == 1:
            cnt_1 += 1
    
    max_val = int(min(cnt_0, cnt_1)/5*4)
    num = int((max_val - max_val%5)/5)

    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
    parameters = {'gamma': np.logspace(5, -15, base = 2, num = 21), 'n_neighbors': np.linspace(5, max_val, num = num, dtype = int)}

    grid = GridSearchCV(Class_KDVM_kknn.KDVM(kernel = 'rbf'), parameters, n_jobs = -1, cv = cv, verbose = 1)
    grid.fit(X, y)
    gamma = grid.best_params_['gamma']
    n_neighbors = grid.best_params_['n_neighbors']

    clf = Class_KDVM_kknn.KDVM(kernel = 'rbf', gamma = gamma, n_neighbors = n_neighbors)
    five_fold = cross_validate(clf, X, y, cv = cv, scoring = 'accuracy', n_jobs = -1)
    mean_ACC = np.mean(five_fold['test_score'])
    print(mean_ACC)

    print(gamma)
    print(n_neighbors)

    '''
    clf = Class_KDVM_kknn.KDVM()
    y_pred = clf.fit_predict(X, y, X)
    ACC = metrics.accuracy_score(y, y_pred)
    print(ACC)
    '''
    '''
    gamma = 0.001953125
    n_neighbors = 200
    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
    clf = Class_KDVM_kknn.KDVM(kernel = 'rbf', gamma = gamma, n_neighbors = n_neighbors)
    five_fold = cross_validate(clf, X, y, cv = cv, scoring = 'accuracy', n_jobs = -1)
    mean_ACC = np.mean(five_fold['test_score'])
    print(mean_ACC)
    '''