import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn import model_selection
from imblearn.metrics import specificity_score
from hsic_kernel_weights_norm import hsic_kernel_weights_norm

def get_feature(file):
    m = np.shape(file)[0]
    n = np.shape(file)[1]
    data = np.zeros((m, n-1))
    for index in range(m):
        data[index] = file[index][1:]
    return data

def get_y_score(y_proba):
    n = np.shape(y_proba)[0]
    temp = np.zeros(n)
    for i in range(n):
        temp[i] = y_proba[i][1]
    return temp

def get_AUPR(y_true, y_score):
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_score, pos_label = 1)
    AUPR = metrics.auc(recall, precision)
    return AUPR


dataset_name = ['Antifp_Main', 'Antifp_DS1', 'Antifp_DS2']
methods_name = ['188-bit', 'AAC', 'ASDC', 'CKSAAP', 'DPC']

G_list_Main = [0.25,8,1,0.25,1]
G_list_DS1 = [0.5,16,1,0.25,0.5]
G_list_DS2 = [0.25,8,0.5,0.25,1]

weight_v_Main = [0.4037,0.1697,0.0472,0.2606,0.1188]
weight_v_DS1 = [0.2074,0,0,0,0.7926]
weight_v_DS2 = [0.2933,0.2258,0.0722,0.2344,0.1744]

for ds in range(2, 3):
    name_ds = dataset_name[ds]
    print('dataset:', name_ds)

    if name_ds == 'Antifp_Main':
        G_list = G_list_Main
        weight_v = weight_v_Main
    elif name_ds == 'Antifp_DS1':
        G_list = G_list_DS1  
        weight_v = weight_v_DS1
    elif name_ds == 'Antifp_DS2':
        G_list = G_list_DS2  
        weight_v = weight_v_DS2

    y_train = np.loadtxt('D:/Study/Bioinformatics/补实验/AFP/feature_matrix/' + name_ds + '/train_label.csv', delimiter = ',')
    y_test = np.loadtxt('D:/Study/Bioinformatics/补实验/AFP/feature_matrix/' + name_ds + '/test_label.csv', delimiter = ',')
    
    n_train = len(y_train)
    n_test = len(y_test)

    n_kernels = 5

    kernel_train_list = []
    kernel_test_list = []
    gram_train = np.zeros((n_train, n_train))
    gram_test = np.zeros((n_test, n_train))

    for it in range(n_kernels):
        name = methods_name[it]
        print(name)
        f1 = np.loadtxt('D:/Study/Bioinformatics/补实验/AFP/feature_matrix/' + name_ds + '/' + name +'/train_' + name +'.csv', delimiter = ',', skiprows = 1)
        f3 = np.loadtxt('D:/Study/Bioinformatics/补实验/AFP/feature_matrix/' + name_ds + '/' + name +'/test_' + name +'.csv', delimiter = ',', skiprows = 1)
        
        X_train = get_feature(f1)
        X_test = get_feature(f3)

        scaler = preprocessing.MinMaxScaler(feature_range = (0, 1)).fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        gram_train = metrics.pairwise.rbf_kernel(X_train, X_train, gamma = G_list[it])
        gram_test = metrics.pairwise.rbf_kernel(X_test, X_train, gamma = G_list[it])
        kernel_train_list.append(gram_train)
        kernel_test_list.append(gram_test)

    for i in range(n_kernels):
        gram_train += kernel_train_list[i]*weight_v[i]
        gram_test += kernel_test_list[i]*weight_v[i]


cv = model_selection.StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)

parameters = {'C': np.logspace(-15, 10, base = 2, num = 26)}
grid = model_selection.GridSearchCV(svm.SVC(kernel = 'precomputed', probability = True), parameters, n_jobs = -1, cv = cv, verbose = 2)
grid.fit(gram_train, y_train)
C = grid.best_params_['C']


clf = svm.SVC(C = C, kernel = 'precomputed', probability = True)

scorerMCC = metrics.make_scorer(metrics.matthews_corrcoef)
scorerSP = metrics.make_scorer(specificity_score)
scorerPR = metrics.make_scorer(metrics.precision_score)
scorerSE = metrics.make_scorer(metrics.recall_score)

scorer = {'ACC':'accuracy', 'recall':scorerSE, 'roc_auc': 'roc_auc', 'MCC':scorerMCC, 'SP':scorerSP}
five_fold = model_selection.cross_validate(clf, gram_train, y_train, cv = cv, scoring = scorer)

mean_ACC = np.mean(five_fold['test_ACC'])
mean_sensitivity = np.mean(five_fold['test_recall'])
mean_AUC = np.mean(five_fold['test_roc_auc'])
mean_MCC = np.mean(five_fold['test_MCC'])
mean_SP = np.mean(five_fold['test_SP'])

print("===========================")
print(weight_v)
print('C =', C)
print("===========================")
#print('five fold:')
print(mean_sensitivity)
print(mean_SP)
print(mean_ACC)
print(mean_MCC)
print(mean_AUC)


clf.fit(gram_train, y_train)

y_score = clf.predict_proba(gram_test)
y_score = get_y_score(y_score)
precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_score)

y_pred = clf.predict(gram_test)
ACC = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
sensitivity = metrics.recall_score(y_test, y_pred)
specificity = specificity_score(y_test, y_pred)
AUC = metrics.roc_auc_score(y_test, y_score)
MCC = metrics.matthews_corrcoef(y_test, y_pred)
AUPR = get_AUPR(y_test, y_score)

#print("===========================")
#print('testing:')
print(sensitivity)
print(specificity)
print(ACC)
print(MCC)
print(AUC)
#print('AUPR =', AUPR)

