import numpy as np
from sklearn import metrics
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, KFold

dataset = ['CKSNAP', 'Kmer4', 'Kmer1234', 'NAC', 'RCKmer', 'DNC', 'TNC']
gamma_list = [2.979166667, 1, 3.5625, 16.89930556, 2.045277778, 12.89583333, 2.670138889]

for i in range(0, 1):
    name = dataset[i]
    print(name)
    f1 = np.loadtxt('D:/Study/Bioinformatics/王浩/data and code/data/feature/mRNA/mRNA_' + name + '.csv',
                    delimiter=',')
    f2 = np.loadtxt('D:/Study/Bioinformatics/王浩/data and code/data/feature/mRNA/mRNA_label.csv', delimiter=',',
                    skiprows=1)
    X = f1
    y = f2

    # scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(X)
    # X = scaler.transform(X)

    cv = KFold(n_splits=10, shuffle=True, random_state=0)

    parameters = {'estimator__C': np.logspace(-10, 10, base=2, num=21)}

    grid = GridSearchCV(OneVsRestClassifier(SVC(kernel='rbf'), n_jobs=-1), parameters, n_jobs=-1,
                        cv=cv, verbose=2)

    grid.fit(X, y)
    C = grid.best_params_['estimator__C']

    # average_pre_score = average_precision_score(Y_test, pre_score_2, average='samples')
    # zero_one_loss_1 = metrics.zero_one_loss(Y_test, pre_y)
    # coverage_error_1 = coverage_error(Y_test, pre_score_2) - 1
    # label_ranking_loss_1 = label_ranking_loss(Y_test, pre_score_2)
    # ham_loss = metrics.hamming_loss(Y_test.T, pre_y.T)
    # acc_score = metrics.accuracy_score(Y_test, pre_y)

    score_zero_one_loss = metrics.make_scorer(metrics.zero_one_loss)
    score_ham_loss = metrics.make_scorer(metrics.hamming_loss)
    score_acc_score = metrics.make_scorer(metrics.accuracy_score)
    average_pre_score = metrics.make_scorer(metrics.average_precision_score, average='samples')

    scorer = {'ACC': score_acc_score, 'zero_one_loss': score_zero_one_loss, 'hamming_loss': score_ham_loss,
              'average_pre_score': average_pre_score}

    clf = OneVsRestClassifier(SVC(kernel='rbf', gamma=gamma_list[i], C=C))

    five_fold = cross_validate(clf, X, y, cv=cv, scoring=scorer, n_jobs=-1)
    mean_ACC = np.mean(five_fold['test_ACC'])
    mean_zero_one_loss = np.mean(five_fold['test_zero_one_loss'])
    mean_hamming_loss = np.mean(five_fold['test_hamming_loss'])
    mean_average_pre_score = np.mean(five_fold['test_average_pre_score'])

    print(C)

    print(mean_ACC)
    print(mean_zero_one_loss)
    print(mean_hamming_loss)
    print(mean_average_pre_score)
