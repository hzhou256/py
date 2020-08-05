import numpy as np
from mRNA import Class_KDVM_knn
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, KFold

file_name = 'human_snoRNA'

dataset = ['CKSNAP', 'Kmer4', 'Kmer1234', 'NAC', 'RCKmer', 'DNC', 'TNC']
for i in range(3, 4):
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

    cv = KFold(n_splits=10, shuffle=True, random_state=0)
    parameters = {'estimator__gamma': np.logspace(5, -15, base=2, num=21),
                  'estimator__n_neighbors': np.linspace(5, 15, num=3, dtype=int)}

    grid = GridSearchCV(OneVsRestClassifier(Class_KDVM_knn.KDVM(kernel='rbf'), n_jobs=-1), parameters, n_jobs=-1,
                        cv=cv, verbose=2)

    grid.fit(X, y)
    gamma = grid.best_params_['estimator__gamma']
    n_neighbors = grid.best_params_['estimator__n_neighbors']

    # average_pre_score = average_precision_score(Y_test, pre_score_2, average='samples')
    # zero_one_loss_1 = metrics.zero_one_loss(Y_test, pre_y)
    # coverage_error_1 = coverage_error(Y_test, pre_score_2) - 1
    # label_ranking_loss_1 = label_ranking_loss(Y_test, pre_score_2)
    # ham_loss = metrics.hamming_loss(Y_test.T, pre_y.T)
    # acc_score = metrics.accuracy_score(Y_test, pre_y)

    score_zero_one_loss = metrics.make_scorer(metrics.zero_one_loss)
    score_ham_loss = metrics.make_scorer(metrics.hamming_loss)
    score_acc_score = metrics.make_scorer(metrics.accuracy_score)
    average_pre_score = metrics.make_scorer(metrics.label_ranking_average_precision_score)

    scorer = {'ACC': score_acc_score, 'zero_one_loss': score_zero_one_loss, 'hamming_loss': score_ham_loss,
              'average_pre_score': average_pre_score}

    clf = OneVsRestClassifier(Class_KDVM_knn.KDVM(kernel='rbf', gamma=gamma, n_neighbors=n_neighbors), n_jobs=-1)
    cv_fold = cross_validate(clf, X, y, cv=cv, scoring=scorer, n_jobs=-1)
    mean_ACC = np.mean(cv_fold['test_ACC'])
    mean_zero_one_loss = np.mean(cv_fold['test_zero_one_loss'])
    mean_hamming_loss = np.mean(cv_fold['test_hamming_loss'])
    mean_average_pre_score = np.mean(cv_fold['test_average_pre_score'])

    print(gamma)
    print(n_neighbors)

    print(mean_ACC)
    print(mean_zero_one_loss)
    print(mean_hamming_loss)
    print(mean_average_pre_score)

    print(cv_fold)
