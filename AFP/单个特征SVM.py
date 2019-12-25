import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn import svm
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
methods_name = ['188-bit', 'AAC', 'ASDC', 'CKSAAP', 'DPC']

C_list_Main = [20.8468516109501, 1.42121073605154, 3.39412848151555, 16.6954828940462, 4.66042447390057]
G_list_Main = [0.0000284792011797527, 76.9918635706838, 170.80198612881, 12.804430034573, 60.6324127241912]
C_list_DS1 = [13.5242595484087, 2.31381991000053, 8.31642346053484, 1.6667283161775, 1.47685707743047]
G_list_DS1 = [0.0000361594727090481, 82.5454327318349, 144.873841101274, 15.1053183447585, 35.778925406486]
C_list_DS2 = [13.1637903107443, 18.9662505387075, 7.46206043541747, 3.53697869125839, 8.0311201079329]
G_list_DS2 = [0.0000786446012462563, 91.4364494328818, 160.34375429646, 9.71484730129853, 47.646239357237]

for ds in range(1,2):
    name_ds = dataset_name[ds]
    print('dataset:', name_ds)
    if name_ds == 'Antifp_Main':
        C_list = C_list_Main
        G_list = G_list_Main
    elif name_ds == 'Antifp_DS1':
        C_list = C_list_DS1
        G_list = G_list_DS1  
    elif name_ds == 'Antifp_DS2':
        C_list = C_list_DS2
        G_list = G_list_DS2  
      
    for it in range(2,3):
        name = methods_name[it]
        print(name)
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
        cost = C_list[it]
        gamma = G_list[it]

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