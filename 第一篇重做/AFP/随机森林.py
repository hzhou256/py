import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import model_selection
from imblearn.metrics import specificity_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

dataset_name = ['Antifp_Main', 'Antifp_DS1', 'Antifp_DS2']
for ds in range(2, 3):
    name_ds = dataset_name[ds]
    print(name_ds)
    methods_name = ['ASDC', 'CKSAAP', 'DPC', 'CAT']
    for it in range(3, 4):
        name = methods_name[it]
        print(name)

        f1 = np.loadtxt('D:/Study/Bioinformatics/补实验/AFP/feature_matrix/' + name_ds + '/' + name + '/train_' + name + '.csv', delimiter = ',')
        f2 = np.loadtxt('D:/Study/Bioinformatics/补实验/AFP/feature_matrix/' + name_ds + '/train_label.csv', delimiter = ',')
        f3 = np.loadtxt('D:/Study/Bioinformatics/补实验/AFP/feature_matrix/' + name_ds + '/' + name + '/test_' + name + '.csv', delimiter = ',')
        f4 = np.loadtxt('D:/Study/Bioinformatics/补实验/AFP/feature_matrix/' + name_ds + '/test_label.csv', delimiter = ',')        

        def get_matrix(file):
            m = np.shape(file)[0]
            n = np.shape(file)[1]
            d = np.zeros((m, n-1))
            for index in range(m):
                d[index] = file[index][1:]
            return d

        X_train = f1
        X_test = f3
        y_train = f2
        y_test = f4

        cv = model_selection.StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)

        # create a function to minimize.
        def RF_accuracy_cv(params, cv = cv, X = X_train, y = y_train):

            # we use this params to create a new LinearSVC Classifier
            model = RandomForestClassifier(**params)
            # and then conduct the cross validation with the same folds as before
            score = -model_selection.cross_val_score(model, X, y, cv = cv, scoring = "accuracy", n_jobs = -1).mean()
            return score

        # possible values of parameters
        space4rf = {
            'max_depth': hp.choice('max_depth', range(1, 100)),
            'n_estimators': hp.choice('n_estimators', range(1, 200)),
            'criterion': hp.choice('criterion', ['gini', 'entropy'])}

        # trials will contain logging information
        trials = Trials()
        best = fmin(fn = RF_accuracy_cv, # function to optimize
                space = space4rf,
                algo = tpe.suggest, # optimization algorithm, hyperotp will select its parameters automatically
                max_evals = 100, # maximum number of iterations
                trials = trials, # logging
                )
        print(best)
        cri = ['gini', 'entropy']
        max_depth = best['max_depth']
        n_estimators = best['n_estimators']
        criterion = cri[best['criterion']]
        print('max_depth =', max_depth)
        print('n_estimators =', n_estimators)
        print('criterion =', criterion)
        clf = RandomForestClassifier(max_depth = max_depth, n_estimators = n_estimators, criterion = criterion)

        #五折交叉验证
        scorerMCC = metrics.make_scorer(metrics.matthews_corrcoef)
        scorerSP = metrics.make_scorer(specificity_score)
        scorerPR = metrics.make_scorer(metrics.precision_score)
        scorerSE = metrics.make_scorer(metrics.recall_score)

        scorer = {'ACC':'accuracy', 'recall':scorerSE, 'roc_auc': 'roc_auc', 'MCC':scorerMCC, 'SP':scorerSP}
        five_fold = model_selection.cross_validate(clf, X_train, y_train, cv = cv, scoring = scorer)

        mean_ACC = np.mean(five_fold['test_ACC'])
        mean_sensitivity = np.mean(five_fold['test_recall'])
        mean_AUC = np.mean(five_fold['test_roc_auc'])
        mean_MCC = np.mean(five_fold['test_MCC'])
        mean_SP = np.mean(five_fold['test_SP'])

        print('five fold:')
        print('SN =', mean_sensitivity)
        print('SP =', mean_SP)
        print('ACC =', mean_ACC)
        print('MCC = ', mean_MCC)
        print('AUC = ', mean_AUC)

        #独立测试集
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        ACC = metrics.accuracy_score(y_test, y_pred)
        sensitivity = metrics.recall_score(y_test, y_pred)
        specificity = specificity_score(y_test, y_pred)
        AUC = metrics.roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
        MCC = metrics.matthews_corrcoef(y_test, y_pred)

        print('Testing set:')
        print('SN =', sensitivity)
        print('SP =', specificity)
        print('ACC =', ACC)
        print('MCC =', MCC)
        print('AUC =', AUC)