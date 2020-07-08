import numpy as np
from sklearn import metrics, svm
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold


dataset = ['wine', 'abalone', 'iris', 'RNA', 'vehicle', 'glass']
for i in range(4, 5):
    name = dataset[i]
    print(name)
    f1 = np.loadtxt('E:/Study/Bioinformatics/UCI/' + name + '/data.csv', delimiter = ',')
    X = f1[:, 0:-1]
    y = f1[:, -1]
    y = np.reshape(y.astype(int), (-1, 1))
    
    scaler = preprocessing.MinMaxScaler(feature_range = (-1, 1)).fit(X)
    X = scaler.transform(X)

    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
    parameters = {'estimator__C': np.logspace(-10, 10, base = 2, num = 21), 'estimator__gamma': np.logspace(5, -15, base = 2, num = 21)}
    grid = GridSearchCV(OneVsRestClassifier(svm.SVC(kernel = 'rbf')), parameters, n_jobs = -1, cv = cv, verbose = 1)
    grid.fit(X, y)
    gamma = grid.best_params_['estimator__gamma']
    C = grid.best_params_['estimator__C'] 

    clf = OneVsRestClassifier(svm.SVC(kernel = 'rbf'))
    five_fold = cross_validate(clf, X, y, cv = cv, scoring = 'accuracy', n_jobs = -1)
    mean_ACC = np.mean(five_fold['test_score'])
    print(mean_ACC)

    print(gamma)
    print(C)




