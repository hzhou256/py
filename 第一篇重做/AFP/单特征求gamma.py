import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
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
methods_name = ['188-bit', 'AAC', 'ASDC', 'CKSAAP', 'DPC']


for ds in range(2, 3):
    name_ds = dataset_name[ds]
    print('dataset:', name_ds)
      
    for it in range(4, 5):
        name = methods_name[it]
        print(name)
        f1 = np.loadtxt('D:/Study/Bioinformatics/补实验/AFP/feature_matrix/' + name_ds + '/' + name +'/train_' + name +'.csv', delimiter = ',', skiprows = 1)
        f2 = np.loadtxt('D:/Study/Bioinformatics/补实验/AFP/feature_matrix/' + name_ds + '/train_label.csv', delimiter = ',')
        f3 = np.loadtxt('D:/Study/Bioinformatics/补实验/AFP/feature_matrix/' + name_ds + '/' + name +'/test_' + name +'.csv', delimiter = ',', skiprows = 1)
        f4 = np.loadtxt('D:/Study/Bioinformatics/补实验/AFP/feature_matrix/' + name_ds + '/test_label.csv', delimiter = ',')

        # np.set_printoptions(suppress = True)
        X_train = get_feature(f1)
        y_train = f2
        X_test = get_feature(f3)
        y_test = f4

        scaler = preprocessing.MinMaxScaler(feature_range = (0, 1)).fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        cv = model_selection.StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)

        parameters = {'gamma': np.logspace(5, -15, base = 2, num = 21), 'C': np.logspace(-15, 10, base = 2, num = 21)}
        grid = model_selection.GridSearchCV(svm.SVC(kernel='rbf'), param_grid=parameters, n_jobs = -1, cv = cv, verbose = 2)
        grid.fit(X_train, y_train)
        cost = grid.best_params_['C']
        gamma = grid.best_params_['gamma']

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
        five_fold = model_selection.cross_validate(clf, X_train, y_train, cv = cv, scoring = scorer, n_jobs=-1)

        mean_ACC = np.mean(five_fold['test_ACC'])
        mean_sensitivity = np.mean(five_fold['test_recall'])
        mean_AUC = np.mean(five_fold['test_roc_auc'])
        mean_MCC = np.mean(five_fold['test_MCC'])
        mean_SP = np.mean(five_fold['test_SP'])

        print("=================================")
        #print('five fold:')
        print(mean_sensitivity)
        print(mean_SP)
        print(mean_ACC)
        print(mean_MCC)
        print(mean_AUC)

        #独立测试集
        y_pred = clf.predict(X_test)
        ACC = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred)
        sensitivity = metrics.recall_score(y_test, y_pred)
        specificity = specificity_score(y_test, y_pred)
        AUC = metrics.roc_auc_score(y_test, clf.decision_function(X_test))
        MCC = metrics.matthews_corrcoef(y_test, y_pred)

        #print('Testing set:')
        print(sensitivity)
        print(specificity)
        print(ACC)
        print(MCC)
        print(AUC)