import numpy as np
import Class_KDVM_knn
from sklearn import metrics
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold


dataset = ['breastw', 'diabetes', 'heart', 'blood']
for i in range(3, 4):
    name = dataset[i]
    print(name)
    f1 = np.loadtxt('E:/Study/Bioinformatics/UCI/' + name + '/data.csv', delimiter = ',')
    X = f1[:, 0:-1]
    y = f1[:, -1]

    scaler = preprocessing.MinMaxScaler(feature_range = (-1, 1)).fit(X)
    X = scaler.transform(X)

    cnt_0, cnt_1 = 0, 0
    for i in range(len(y)):
        if y[i] == -1:
            y[i] = 0
            cnt_0 += 1
        elif y[i] == 1:
            cnt_1 += 1

    max_val = int(min(cnt_0, cnt_1)/5*4)
    num = int((max_val - max_val%5)/5)
    '''
    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
    parameters = {'gamma': np.logspace(5, -15, base = 2, num = 21), 'n_neighbors': np.linspace(5, max_val, num = num, dtype = int)}

    grid = GridSearchCV(Class_KDVM_knn.KDVM(kernel = 'rbf'), parameters, n_jobs = -1, cv = cv, verbose = 1)
    grid.fit(X, y)
    gamma = grid.best_params_['gamma']
    n_neighbors = grid.best_params_['n_neighbors']

    clf = Class_KDVM_knn.KDVM(kernel = 'rbf', gamma = gamma, n_neighbors = n_neighbors)
    five_fold = cross_validate(clf, X, y, cv = cv, scoring = 'accuracy', n_jobs = -1)
    mean_ACC = np.mean(five_fold['test_score'])
    print(mean_ACC)

    print(gamma)
    print(n_neighbors)
    '''
    '''
    clf = Class_KDVM_knn.KDVM()
    y_pred = clf.fit_predict(X, y, X)
    ACC = metrics.accuracy_score(y, y_pred)
    print(ACC)
    '''
    '''
    gamma = 0.0039
    n_neighbors = 40
    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
    clf = Class_KDVM_knn.KDVM(kernel = 'rbf', gamma = gamma, n_neighbors = n_neighbors)
    five_fold = cross_validate(clf, X, y, cv = cv, scoring = 'accuracy', n_jobs = -1)
    mean_ACC = np.mean(five_fold['test_score'])
    print(mean_ACC)
    '''

    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
    parameters_1 = {'n_neighbors': np.linspace(5, max_val, num = num, dtype = int)}

    grid_1 = GridSearchCV(Class_KDVM_knn.KDVM(kernel = 'rbf'), parameters_1, n_jobs = -1, cv = cv, verbose = 2)
    grid_1.fit(X, y)
    
    n_neighbors = grid_1.best_params_['n_neighbors']

    parameters_2 = {'gamma': np.logspace(5, -15, base = 2, num = 21)}
    grid_2 = GridSearchCV(Class_KDVM_knn.KDVM(kernel = 'rbf', n_neighbors = n_neighbors), parameters_2, n_jobs = -1, cv = cv, verbose = 2)
    grid_2.fit(X, y)

    gamma = grid_2.best_params_['gamma']

    clf = Class_KDVM_knn.KDVM(kernel = 'rbf', gamma = gamma, n_neighbors = n_neighbors)
    five_fold = cross_validate(clf, X, y, cv = cv, scoring = 'accuracy', n_jobs = -1)
    mean_ACC = np.mean(five_fold['test_score'])
    print(mean_ACC)

    print(gamma)
    print(n_neighbors)

    '''
    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
    parameters = {'C': np.logspace(-10, 10, base = 2, num = 21), 'gamma': np.logspace(5, -15, base = 2, num = 21)}
    grid = GridSearchCV(svm.SVC(kernel = 'rbf'), parameters, n_jobs = -1, cv = cv, verbose = 2)
    grid.fit(X, y)
    gamma = grid.best_params_['gamma']
    C = grid.best_params_['C'] 

    clf = svm.SVC(C = C, gamma = gamma, kernel = 'rbf', probability = True)
    five_fold = cross_validate(clf, X, y, cv = cv, scoring = 'accuracy', n_jobs = -1)
    mean_ACC = np.mean(five_fold['test_score'])
    print(mean_ACC)

    print(gamma)
    print(C)
    '''
