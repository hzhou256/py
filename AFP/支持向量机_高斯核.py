import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection
from imblearn.metrics import specificity_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


def get_feature(file):
    m = np.shape(file)[0]
    n = np.shape(file)[1]
    data = np.zeros((m, n-1))
    for index in range(m):
        data[index] = file[index][1:]
    return data


dataset_name = ['Antifp_Main', 'Antifp_DS1', 'Antifp_DS2']
methods_name = ['ASDC', 'CKSAAP', 'DPC', 'CAT']

for ds in range(3):
    name_ds = dataset_name[ds]
    print(name_ds)
    for it in range(0, 1):
        name = methods_name[it]
        print(name)
        f1 = np.loadtxt('E:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds + '/' + name +'/train_' + name +'.csv', delimiter = ',', skiprows = 1)
        f2 = np.loadtxt('E:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds + '/train_label.csv', delimiter = ',')
        f3 = np.loadtxt('E:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds + '/' + name +'/test_' + name +'.csv', delimiter = ',', skiprows = 1)
        f4 = np.loadtxt('E:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds + '/test_label.csv', delimiter = ',')

        np.set_printoptions(suppress = True)
        X_train = get_feature(f1)
        y_train = f2
        X_test = get_feature(f3)
        y_test = f4

        cv = model_selection.StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
        # create a function to minimize.
        def SVM_accuracy_cv(params, cv = cv, X = X_train, y = y_train):
            # the function gets a set of variable parameters in "param"
            params = {'C': params['C'], 'gamma': params['gamma']}
            # we use this params to create a new LinearSVC Classifier
            model = svm.SVC(kernel = 'rbf', probability = True, **params)
            # and then conduct the cross validation with the same folds as before
            score = -model_selection.cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs = -1).mean()
            return score

        # possible values of parameters
        space= {'C': hp.loguniform('C', low = np.log(1e-7) , high = np.log(1e5)), 
                'gamma': hp.loguniform('gamma', low = np.log(1e-7) , high = np.log(1e5))}

        # trials will contain logging information
        trials = Trials()
        best = fmin(fn = SVM_accuracy_cv, # function to optimize
                space = space,
                algo = tpe.suggest, # optimization algorithm, hyperotp will select its parameters automatically
                max_evals = 100, # maximum number of iterations
                trials = trials, # logging
                )

        cost = best['C']
        gamma = best['gamma']

        print('C =', cost)
        print('g =', gamma)

        clf = svm.SVC(C = cost, gamma = gamma, kernel = 'rbf', probability = True)
        clf.fit(X_train, y_train)

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
        y_pred = clf.predict(X_test)
        ACC = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred)
        sensitivity = metrics.recall_score(y_test, y_pred)
        specificity = specificity_score(y_test, y_pred)
        AUC = metrics.roc_auc_score(y_test, clf.decision_function(X_test))
        MCC = metrics.matthews_corrcoef(y_test, y_pred)

        print('Testing set:')
        print('SN =', sensitivity)
        print('SP =', specificity)
        print('ACC =', ACC)
        print('MCC =', MCC)
        print('AUC =', AUC)        
