import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, KFold

dataset = ['CKSNAP', 'Kmer4', 'Kmer1234', 'NAC', 'RCKmer', 'DNC', 'TNC']
for i in range(0, 1):
    name = dataset[i]
    print(name)

    gram = np.loadtxt('D:/Study/Bioinformatics/mRNA/rbf_kernel/KM_rbf_' + name + '_train.csv',
                      delimiter=',')
    y = np.loadtxt('D:/Study/Bioinformatics/王浩/data and code/data/feature/mRNA/mRNA_label.csv', delimiter=',',
                   skiprows=1)

    print(gram)

    cv = KFold(n_splits=10, shuffle=True, random_state=0)
    parameters = {'estimator__C': np.logspace(-15, 10, base=2, num=42)}
    grid = GridSearchCV(OneVsRestClassifier(SVC(kernel='precomputed'), n_jobs=-1), parameters, n_jobs=-1,
                        cv=cv, verbose=2)
    grid.fit(gram, y)
    C = grid.best_params_['estimator__C']

    score_zero_one_loss = metrics.make_scorer(metrics.zero_one_loss)
    score_ham_loss = metrics.make_scorer(metrics.hamming_loss)
    score_acc_score = metrics.make_scorer(metrics.accuracy_score)
    average_pre_score = metrics.make_scorer(metrics.average_precision_score, average='samples')

    scorer = {'ACC': score_acc_score, 'zero_one_loss': score_zero_one_loss, 'hamming_loss': score_ham_loss,
              'average_pre_score': average_pre_score}

    clf = OneVsRestClassifier(SVC(kernel='precomputed', C=C))

    cvTest = cross_validate(clf, gram, y, cv=cv, scoring=scorer, n_jobs=-1)
    mean_ACC = np.mean(cvTest['test_ACC'])
    mean_zero_one_loss = np.mean(cvTest['test_zero_one_loss'])
    mean_hamming_loss = np.mean(cvTest['test_hamming_loss'])
    mean_average_pre_score = np.mean(cvTest['test_average_pre_score'])

    print(C)

    print(mean_ACC)
    print(mean_zero_one_loss)
    print(mean_hamming_loss)
    print(mean_average_pre_score)
