import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn import model_selection
from imblearn.metrics import specificity_score
from hsic_kernel_weights_norm import hsic_kernel_weights_norm


# remove first row of feature csv files
def get_feature(file):
    m = np.shape(file)[0]
    n = np.shape(file)[1]
    data = np.zeros((m, n-1))
    for index in range(m):
        data[index] = file[index][1:]
    return data

# get real value score
def get_y_score(y_proba):
    n = np.shape(y_proba)[0]
    temp = np.zeros(n)
    for i in range(n):
        temp[i] = y_proba[i][1]
    return temp

# calculating AUPR
def get_AUPR(y_true, y_score):
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_score, pos_label = 1)
    AUPR = metrics.auc(recall, precision)
    return AUPR


dataset_name = ['Antifp_Main', 'Antifp_DS1', 'Antifp_DS2']
methods_name = ['188-bit', 'AAC', 'ASDC', 'CKSAAP', 'DPC']

# gamma for each feature on three datasets
G_list_Main = [0.25,8,1,0.25,1]
G_list_DS1 = [0.5,16,1,0.25,0.5]
G_list_DS2 = [0.25,8,0.5,0.25,1]

# cost parameter for three datasets
c_list = [4,4,1]

# kernel weights calculated by HSIC for three datasets
weight_Main = [0.20779853,0.21165743,0.20338872,0.18990633,0.18724899]
weight_DS1 = [0.20596246,0.20075947,0.20297282,0.19504831,0.19525695]
weight_DS2 = [0.20454586,0.23384408,0.22289349,0.1704432,0.16827337]

for ds in range(0, 1): # select datasets using iteration
    name_ds = dataset_name[ds]
    print('dataset:', name_ds)

    if name_ds == 'Antifp_Main':
        G_list = G_list_Main
        weight_v = weight_Main
    elif name_ds == 'Antifp_DS1':
        G_list = G_list_DS1  
        weight_v = weight_DS1
    elif name_ds == 'Antifp_DS2':
        G_list = G_list_DS2 
        weight_v = weight_DS2 

    # path for train and test labels
    y_train = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds + '/train_label.csv', delimiter = ',') 
    y_test = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds + '/test_label.csv', delimiter = ',')
    
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
        # path for extracted features
        f1 = np.loadtxt('D:/Study/Bioinformatics/补实验/AFP/feature_matrix/' + name_ds + '/' + name +'/train_' + name +'.csv', delimiter = ',', skiprows = 1)
        f3 = np.loadtxt('D:/Study/Bioinformatics/补实验/AFP/feature_matrix/' + name_ds + '/' + name +'/test_' + name +'.csv', delimiter = ',', skiprows = 1)
        
        X_train = get_feature(f1)
        X_test = get_feature(f3)

        # scale features into (0, 1)
        scaler = preprocessing.MinMaxScaler(feature_range = (0, 1)).fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # calculating rbf kernels
        gram_train = metrics.pairwise.rbf_kernel(X_train, X_train, gamma = G_list[it])
        gram_test = metrics.pairwise.rbf_kernel(X_test, X_train, gamma = G_list[it])
        kernel_train_list.append(gram_train)
        kernel_test_list.append(gram_test)

    # weight_v = hsic_kernel_weights_norm(kernel_train_list, y_train, 1, 0.01, 0) # calculating weights using HSIC during experiment

    # combine kernels
    for i in range(n_kernels):
        gram_train += kernel_train_list[i]*weight_v[i]
        gram_test += kernel_test_list[i]*weight_v[i]

# five-fold cross validation
cv = model_selection.StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)

# cost parameter for SVM
C = c_list[ds]

# init SVM classifier with precomputed kernel
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

print('five fold:')
print(mean_sensitivity)
print(mean_SP)
print(mean_ACC)
print(mean_MCC)
print(mean_AUC)

# fit SVM
clf.fit(gram_train, y_train)

y_score = clf.predict_proba(gram_test)
y_score = get_y_score(y_score)
precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_score)

# predict testing samples
y_pred = clf.predict(gram_test)

ACC = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
sensitivity = metrics.recall_score(y_test, y_pred)
specificity = specificity_score(y_test, y_pred)
AUC = metrics.roc_auc_score(y_test, y_score)
MCC = metrics.matthews_corrcoef(y_test, y_pred)
AUPR = get_AUPR(y_test, y_score)

print("===========================")
print('testing:')
print(sensitivity)
print(specificity)
print(ACC)
print(MCC)
print(AUC)
#print('AUPR =', AUPR)

