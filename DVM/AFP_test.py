import collections
import numpy as np
import Class_KDVM_knn
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold



def get_feature(file):
    m = np.shape(file)[0]
    n = np.shape(file)[1]
    data = np.zeros((m, n-1))
    for index in range(m):
        data[index] = file[index][1:]
    return data


dataset_name = ['Antifp_Main', 'Antifp_DS1', 'Antifp_DS2']
methods_name = ['ASDC', 'CKSAAP', 'DPC']

for ds in range(0, 1):
    name_ds = dataset_name[ds]
    print(name_ds)
    for it in range(0, 1):
        name = methods_name[it]
        print(name)
        f1 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds + '/' + name +'/train_' + name +'.csv', delimiter = ',', skiprows = 1)
        f2 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds + '/train_label.csv', delimiter = ',')
        f3 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds + '/' + name +'/test_' + name +'.csv', delimiter = ',', skiprows = 1)
        f4 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds + '/test_label.csv', delimiter = ',')

        np.set_printoptions(suppress = True)
        X_train = get_feature(f1)
        y_train = f2
        X_test = get_feature(f3)
        y_test = f4

        scaler = preprocessing.MinMaxScaler(feature_range = (-1, 1)).fit(X_train)
        X_train = scaler.transform(X_train)
        scaler = preprocessing.MinMaxScaler(feature_range = (-1, 1)).fit(X_test)
        X_test = scaler.transform(X_test)

        cnt = dict(collections.Counter(y_train))
        n_class = len(cnt)
        max_val = int(min(cnt.values())/5*4)
        num = int((max_val - max_val%5)/5)

        cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
        parameters = {'gamma': np.logspace(5, -15, base = 2, num = 21), 'n_neighbors': np.linspace(210, 300, num = 10, dtype = int)}
        
        grid = GridSearchCV(Class_KDVM_knn.KDVM(kernel = 'rbf'), parameters, n_jobs = -1, cv = cv, verbose = 2)
        grid.fit(X_train, y_train)
        gamma = grid.best_params_['gamma']
        n_neighbors = grid.best_params_['n_neighbors']
        
        
        #gamma = 0.03125
        #n_neighbors = 90

        clf = Class_KDVM_knn.KDVM(kernel = 'rbf', gamma = gamma, n_neighbors = n_neighbors)
        five_fold = cross_validate(clf, X_train, y_train, cv = cv, scoring = 'accuracy', n_jobs = -1)
        mean_ACC = np.mean(five_fold['test_score'])
        print(mean_ACC)

        print(gamma)
        print(n_neighbors)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        ACC = metrics.accuracy_score(y_test, y_pred)
        print(ACC)
