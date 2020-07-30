import collections
import numpy as np
import Class_KDVM_knn
from sklearn import metrics
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, KFold


dataset = ['CKSNAP', 'DNC', 'Kmer4', 'Kmer1234', 'NAC', 'RCKmer', 'TNC']
for i in range(0, 1):
    name = dataset[i]
    print(name)
    f1 = np.loadtxt('D:/Study/Bioinformatics/王浩/data and code/data/feature/mRNA/mRNA_'+name+'.csv', delimiter = ',')
    f2 = np.loadtxt('D:/Study/Bioinformatics/王浩/data and code/data/feature/mRNA/mRNA_label.csv', delimiter = ',', skiprows = 1)
    X = f1
    y = f2
    
    
    scaler = preprocessing.MinMaxScaler(feature_range = (-1, 1)).fit(X)
    X = scaler.transform(X)

    
    cv = KFold(n_splits = 5, shuffle = True, random_state = 0)
    parameters = {'estimator__gamma': np.logspace(5, -15, base = 2, num = 21), 'estimator__n_neighbors': np.linspace(10, 110, num = 11, dtype = int)}
    
    
    grid = GridSearchCV(OneVsRestClassifier(Class_KDVM_knn.KDVM(kernel = 'rbf'), n_jobs = -1), parameters, n_jobs = -1, cv = cv, verbose = 2)
    grid.fit(X, y)
    gamma = grid.best_params_['estimator__gamma']
    n_neighbors = grid.best_params_['estimator__n_neighbors']


    #average_pre_score = average_precision_score(Y_test, pre_score_2, average='samples')
    #zero_one_loss_1 = metrics.zero_one_loss(Y_test, pre_y)
    #coverage_error_1 = coverage_error(Y_test, pre_score_2) - 1
    #label_ranking_loss_1 = label_ranking_loss(Y_test, pre_score_2)
    #ham_loss = metrics.hamming_loss(Y_test.T, pre_y.T)
    #acc_score = metrics.accuracy_score(Y_test, pre_y)

    score_zero_one_loss = metrics.make_scorer(metrics.zero_one_loss)
    score_ham_loss = metrics.make_scorer(metrics.hamming_loss)
    score_acc_score = metrics.make_scorer(metrics.accuracy_score)

    scorer = {'ACC':score_acc_score, 'zero_one_loss':score_zero_one_loss, 'hamming_loss':score_ham_loss}

    clf = OneVsRestClassifier(Class_KDVM_knn.KDVM(kernel = 'rbf', gamma = gamma, n_neighbors = n_neighbors), n_jobs = -1)
    five_fold = cross_validate(clf, X, y, cv = cv, scoring = scorer, n_jobs = -1)
    mean_ACC = np.mean(five_fold['test_ACC'])
    mean_zero_one_loss = np.mean(five_fold['test_zero_one_loss'])
    mean_hamming_losss = np.mean(five_fold['test_hamming_loss'])

    print(gamma)
    print(n_neighbors)

    print(mean_ACC)
    print(mean_zero_one_loss)
    print(mean_hamming_losss)


    




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
