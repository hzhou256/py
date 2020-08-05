import numpy as np
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, KFold

from mRNA import Class_KDVM_multi

file_name = 'snoRNA'

dataset = ['CKSNAP', 'Kmer4', 'Kmer1234', 'NAC', 'RCKmer', 'DNC', 'TNC']
for i in range(0, 7):
    name = dataset[i]
    print(name)

    f1 = np.loadtxt(
        'D:/Study/Bioinformatics/王浩/data and code/data/feature/' + file_name + '/' + file_name + '_' + name + '.csv',
        delimiter=',')
    f2 = np.loadtxt(
        'D:/Study/Bioinformatics/王浩/data and code/data/feature/' + file_name + '/' + file_name + '_label.csv',
        delimiter=',', skiprows=1)

    X = f1
    y = f2

    score_zero_one_loss = metrics.make_scorer(metrics.zero_one_loss)
    score_ham_loss = metrics.make_scorer(metrics.hamming_loss)
    score_acc_score = metrics.make_scorer(metrics.accuracy_score)
    average_pre_score = metrics.make_scorer(metrics.label_ranking_average_precision_score)

    # parameters = {'gamma': np.logspace(5, -15, base=2, num=21), 'n_neighbors': np.linspace(5, 15, num=3, dtype=int)}
    cv = KFold(n_splits=10, shuffle=True, random_state=True)
    '''
    grid = GridSearchCV(Class_KDVM_multi.multilabel_KDVM(kernel='rbf'), parameters, scoring=average_pre_score,
                        n_jobs=-1, cv=cv, verbose=0)

    grid.fit(X, y)
    gamma = grid.best_params_['gamma']
    n_neighbors = grid.best_params_['n_neighbors']
    '''

    best_ACC = 0
    best_ham = 0
    best_zero = 0
    best_AP = 0
    best_gamma = 0

    for g in np.logspace(5, -15, base=2, num=21):
        # print(g)
        gamma = g
        n_neighbors = 15

        ACC_score = []
        ham_loss_score = []
        zero_one_loss_score = []
        AP_score = []

        for train_index, test_index in cv.split(X, y):
            # print('train_index', train_index, 'test_index', test_index)
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]

            clf = Class_KDVM_multi.multilabel_KDVM(kernel='rbf', gamma=gamma, n_neighbors=n_neighbors)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)

            AP = metrics.label_ranking_average_precision_score(y_test, y_prob)
            ham_loss = metrics.hamming_loss(y_test, y_pred)
            zero_one_loss = metrics.zero_one_loss(y_test, y_pred)
            ACC = metrics.accuracy_score(y_test, y_pred)

            AP_score.append(AP)
            ham_loss_score.append(ham_loss)
            zero_one_loss_score.append(zero_one_loss)
            ACC_score.append(ACC)

        if best_ACC < np.mean(ACC_score):
            best_ACC = np.mean(ACC_score)
            best_zero = np.mean(zero_one_loss_score)
            best_ham = np.mean(ham_loss_score)
            best_AP = np.mean(AP_score)
            best_gamma = gamma
    
    print("=====================")
    print(best_gamma)
    print(best_ACC)
    print(best_zero)
    print(best_ham)
    print(best_AP)
