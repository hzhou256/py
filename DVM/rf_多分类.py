import numpy as np
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold


dataset = ['wine', 'abalone', 'iris', 'RNA', 'vehicle', 'glass']
for i in range(2, 3):
    name = dataset[i]
    print(name)
    f1 = np.loadtxt('D:/Study/Bioinformatics/UCI/' + name + '/data.csv', delimiter = ',')
    X = f1[:, 0:-1]
    y = f1[:, -1]
    #y = np.reshape(y.astype(int), (-1, 1))
    
    scaler = preprocessing.MinMaxScaler(feature_range = (-1, 1)).fit(X)
    X = scaler.transform(X)

    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
    parameters_1 = {'n_estimators': np.arange(1, 200)}
    parameters_2 = {'max_depth': np.arange(1, 100), 'criterion': ['gini', 'entropy']}

    grid_1 = GridSearchCV(RandomForestClassifier(), parameters_1, n_jobs = -1, cv = cv, verbose = 1)
    grid_1.fit(X, y)

    n_estimators = grid_1.best_params_['n_estimators']

    grid_2 = GridSearchCV(RandomForestClassifier(n_estimators = n_estimators), parameters_2, n_jobs = -1, cv = cv, verbose = 1)
    grid_2.fit(X, y)

    max_depth = grid_2.best_params_['max_depth']
    criterion = grid_2.best_params_['criterion']

    clf = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, criterion = criterion)

    five_fold = cross_validate(clf, X, y, cv = cv, scoring = 'accuracy', n_jobs = -1)
    mean_ACC = np.mean(five_fold['test_score'])
    print(mean_ACC)

    print('n_estimators = ', n_estimators)
    print('max_depth = ', max_depth)
    print('criterion = ', criterion)



