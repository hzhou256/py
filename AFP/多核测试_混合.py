import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection
from imblearn.metrics import specificity_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


dataset_name = ['Antifp_Main', 'Antifp_DS1', 'Antifp_DS2']
for ds in range(2,3):
    name_ds = dataset_name[ds]
    print('dataset:', name_ds)
    methods_name = ['HSIC', 'FKL', 'TKA', 'HKA', 'CKA']
    for it in range(1):
        name = methods_name[it]
        print(name + ':')

        f1 = np.loadtxt('D:/Study/Bioinformatics/AFP/kernel_matrix_G5+T2/' + name_ds + '/KM_train/combine_' + name + '_train.csv', delimiter = ',')
        f2 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds + '/train_label.csv', delimiter = ',')
        f3 = np.loadtxt('D:/Study/Bioinformatics/AFP/kernel_matrix_G5+T2/' + name_ds + '/KM_test/combine_' + name + '_test.csv', delimiter = ',')
        f4 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds + '/test_label.csv', delimiter = ',')

        np.set_printoptions(suppress = True)
        gram = f1
        y_train = f2
        gram_test = f3
        y_test = f4

        cv = model_selection.StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
        #parameters = {'C': np.logspace(-5, 2, base = 10)}
        #grid = model_selection.GridSearchCV(svm.SVC(kernel = 'precomputed', probability = True), parameters, n_jobs = -1, cv = cv)
        #grid.fit(gram, y_train)
        #cost = grid.best_params_['C']
        #print('C =', cost)

        # create a function to minimize.
        def SVM_accuracy_cv(params, cv = cv, X = gram, y = y_train):
            # the function gets a set of variable parameters in "param"
            params = {'C': params['C']}
            # we use this params to create a new LinearSVC Classifier
            model = svm.SVC(kernel = 'precomputed', probability = True, **params)
            # and then conduct the cross validation with the same folds as before
            score = -model_selection.cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs = -1).mean()
            return score

        # possible values of parameters
        space= {'C': hp.loguniform('C', low = np.log(1e-7) , high = np.log(1e2))}

        # trials will contain logging information
        trials = Trials()
        best = fmin(fn = SVM_accuracy_cv, # function to optimize
                space = space,
                algo = tpe.suggest, # optimization algorithm, hyperotp will select its parameters automatically
                max_evals = 200, # maximum number of iterations
                trials = trials, # logging
                )
        cost = best['C']
        print('C =', cost)
        clf = svm.SVC(C = cost, kernel = 'precomputed', probability = True)

        #五折交叉验证
        scorerMCC = metrics.make_scorer(metrics.matthews_corrcoef)
        scorerSP = metrics.make_scorer(specificity_score)
        scorerPR = metrics.make_scorer(metrics.precision_score)
        scorerSE = metrics.make_scorer(metrics.recall_score)

        scorer = {'ACC':'accuracy', 'recall':scorerSE, 'roc_auc': 'roc_auc', 'MCC':scorerMCC, 'SP':scorerSP}
        five_fold = model_selection.cross_validate(clf, gram, y_train, cv = cv, scoring = scorer)

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
        clf.fit(gram, y_train)

        y_pred = clf.predict(gram_test)

        ACC = metrics.accuracy_score(y_test, y_pred)
        sensitivity = metrics.recall_score(y_test, y_pred)
        specificity = specificity_score(y_test, y_pred)
        AUC = metrics.roc_auc_score(y_test, clf.decision_function(gram_test))
        MCC = metrics.matthews_corrcoef(y_test, y_pred)

        print('Testing set:')
        print('SN =', sensitivity)
        print('SP =', specificity)
        print('ACC =', ACC)
        print('MCC =', MCC)
        print('AUC =', AUC)