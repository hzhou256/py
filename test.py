import warnings
warnings.filterwarnings("ignore")
import numpy as np
from thundersvm import SVC
from sklearn import metrics
from sklearn import model_selection
from imblearn.metrics import specificity_score


def get_feature(file):
    m = np.shape(file)[0]
    n = np.shape(file)[1]
    data = np.zeros((m, n-1))
    for index in range(m):
        data[index] = file[index][1:]
    return data

dataset_name = ['Antifp_Main', 'Antifp_DS1', 'Antifp_DS2']
methods_name = ['188-bit', 'AAC', 'ASDC', 'CKSAAP', 'CTD', 'DPC']

for ds in range(3):
    name_ds = dataset_name[ds]
    print('dataset:', name_ds)  
    for it in range(6):
        name = methods_name[it]

        f1 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds + '/' + name +'/train_' + name +'.csv', delimiter = ',', skiprows = 1)
        f2 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds + '/train_label.csv', delimiter = ',')
        f3 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds + '/' + name +'/test_' + name +'.csv', delimiter = ',', skiprows = 1)
        f4 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds + '/test_label.csv', delimiter = ',')

        np.set_printoptions(suppress = True)
        X_train = get_feature(f1)
        y_train = f2
        X_test = get_feature(f3)
        y_test = f4

        cv = model_selection.StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)

        parameters = {'C':np.logspace(-8, 5, base = 2), 'gamma':np.logspace(8, -8, base = 2)}
        grid = model_selection.GridSearchCV(SVC(kernel = 'rbf', probability = True, gpu_id = 0), parameters, n_jobs = -1)
        cost = grid.best_params_['C']
        gamma = grid.best_params_['gamma']

        print('G =', cost)
        print('C =', gamma)

        clf = SVC(C = cost, gamma = gamma, kernel = 'rbf', probability = True, gpu_id = 0)
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