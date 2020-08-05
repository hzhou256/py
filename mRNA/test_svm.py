import numpy as np
from sklearn import metrics
# from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, KFold

file_name = "human_snoRNA"
dataset = ['CKSNAP', 'Kmer4', 'Kmer1234', 'NAC', 'RCKmer', 'DNC', 'TNC']

gamma_list_snoRNA = [2.210416667, 0.0375, 1.566964286, 3.971875, 3.18375, 2.459375, 2.0025]
gamma_list_mRNA = [2.979166667, 1, 3.5625, 16.89930556, 2.045277778, 12.89583333, 2.670138889]
gamma_list_human_snoRNA = [0.0375, 0.110416667, 0.090625, 2.47375, 2.072916667, 2.065178571, 1.8390625]

if file_name == "mRNA":
    gamma_list = gamma_list_mRNA
elif file_name == "snoRNA":
    gamma_list = gamma_list_snoRNA
elif file_name == "human_snoRNA":
    gamma_list = gamma_list_human_snoRNA

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

    # scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(X)
    # X = scaler.transform(X)

    score_zero_one_loss = metrics.make_scorer(metrics.zero_one_loss)
    score_ham_loss = metrics.make_scorer(metrics.hamming_loss)
    score_acc_score = metrics.make_scorer(metrics.accuracy_score)
    average_pre_score = metrics.make_scorer(metrics.label_ranking_average_precision_score)

    cv = KFold(n_splits=10, shuffle=True, random_state=True)

    # parameters = {'estimator__C': np.logspace(-15, 10, base=2, num=26)}
    parameters = {'estimator__C': np.logspace(-15, 10, base=2, num=26),
                  'estimator__gamma': np.logspace(10, -15, base=2, num=26)}

    grid = GridSearchCV(OneVsRestClassifier(SVC(kernel='rbf'), n_jobs=-1), parameters, n_jobs=-1,
                        cv=cv, verbose=0, scoring=average_pre_score)

    grid.fit(X, y)
    C = grid.best_params_['estimator__C']
    gamma = grid.best_params_['estimator__gamma']

    # C = 4.5

    # average_pre_score = average_precision_score(Y_test, pre_score_2, average='samples')
    # zero_one_loss_1 = metrics.zero_one_loss(Y_test, pre_y)
    # coverage_error_1 = coverage_error(Y_test, pre_score_2) - 1
    # label_ranking_loss_1 = label_ranking_loss(Y_test, pre_score_2)
    # ham_loss = metrics.hamming_loss(Y_test.T, pre_y.T)
    # acc_score = metrics.accuracy_score(Y_test, pre_y)

    scorer = {'ACC': score_acc_score, 'zero_one_loss': score_zero_one_loss, 'hamming_loss': score_ham_loss,
              'average_pre_score': average_pre_score}

    # clf = OneVsRestClassifier(SVC(kernel='rbf', gamma=gamma_list[i], C=C))
    clf = OneVsRestClassifier(SVC(kernel='rbf', gamma=gamma, C=C))

    five_fold = cross_validate(clf, X, y, cv=cv, scoring=scorer, n_jobs=-1)
    mean_ACC = np.mean(five_fold['test_ACC'])
    mean_zero_one_loss = np.mean(five_fold['test_zero_one_loss'])
    mean_hamming_loss = np.mean(five_fold['test_hamming_loss'])
    mean_average_pre_score = np.mean(five_fold['test_average_pre_score'])

    print(C)
    print(gamma)
    print("====================")

    print(mean_ACC)
    print(mean_zero_one_loss)
    print(mean_hamming_loss)
    print(mean_average_pre_score)
