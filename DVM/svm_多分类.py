import numpy as np
from sklearn import metrics, svm
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold


dataset = ['wine', 'abalone', 'iris', 'RNA', 'vehicle', 'glass']
for i in range(5, 6):
    name = dataset[i]
    print(name)
    f1 = np.loadtxt('D:/Study/Bioinformatics/UCI/' + name + '/data.csv', delimiter = ',')
    X = f1[:, 0:-1]
    y = f1[:, -1]
    #y = np.reshape(y.astype(int), (-1, 1))
    
    scaler = preprocessing.MinMaxScaler(feature_range = (-1, 1)).fit(X)
    X = scaler.transform(X)

    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
    parameters = {'C': np.logspace(-5, 15, base = 2, num = 21), 'gamma': np.logspace(3, -15, base = 2, num = 19)}
    grid = GridSearchCV(svm.SVC(kernel = 'rbf', decision_function_shape = 'ovo'), parameters, n_jobs = -1, cv = cv, verbose = 1)
    grid.fit(X, y)
    gamma = grid.best_params_['gamma']
    C = grid.best_params_['C']

    clf = svm.SVC(kernel = 'rbf', gamma = gamma, C = C)
    five_fold = cross_validate(clf, X, y, cv = cv, scoring = 'accuracy', n_jobs = -1)
    mean_ACC = np.mean(five_fold['test_score'])
    print(mean_ACC)

    print(gamma)
    print(C)




