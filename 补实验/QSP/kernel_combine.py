import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection
from imblearn.metrics import specificity_score
from hsic_kernel_weights_norm import hsic_kernel_weights_norm


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


y_train = np.loadtxt('D:/Study/Bioinformatics/补实验/QSP/features/train_label.csv', delimiter = ',')
#y_test = np.loadtxt('D:/Study/Bioinformatics/补实验/QSP/features/test_label.csv', delimiter = ',')
n_train = len(y_train)
#n_test = len(y_test)


kernel_train_list = []
#kernel_test_list = []
gram_train = np.zeros((n_train, n_train))
#gram_test = np.zeros((n_test, n_train))

n_kernels = 3

methods_name = ['188-bit', 'ASDC', 'CKSAAP', 'DPC', 'AAC']
for it in range(n_kernels):
    name = methods_name[it]
    gram_train = np.loadtxt('D:/Study/Bioinformatics/补实验/QSP/kernels/K_train_'+name+'.csv', delimiter = ',')
    #gram_test = np.loadtxt('D:/Study/Bioinformatics/补实验/QSP/kernels/K_test_'+name+'.csv', delimiter = ',')
    kernel_train_list.append(gram_train)
    #kernel_test_list.append(gram_test)

weight_v = hsic_kernel_weights_norm(kernel_train_list, y_train, 1, 0.01, 0.001)

for i in range(n_kernels):
    gram_train += kernel_train_list[i]*weight_v[i]
    #gram_test += kernel_test_list[i]*weight_v[i]


cv = model_selection.StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)

parameters = {'C': np.logspace(-15, 10, base = 2, num = 52)}
grid = model_selection.GridSearchCV(svm.SVC(kernel = 'precomputed', probability = True), parameters, n_jobs = -1, cv = cv, verbose = 2)
grid.fit(gram_train, y_train)
C = grid.best_params_['C']
print('C =', C)


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

print(weight_v)
#print('five fold:')
print(mean_sensitivity)
print(mean_SP)
print(mean_ACC)
print(mean_MCC)
print(mean_AUC)

