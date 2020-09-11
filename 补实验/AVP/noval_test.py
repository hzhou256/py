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


y_train = np.loadtxt('D:/Study/Bioinformatics/补实验/AVP/features/train_label.csv', delimiter = ',')
y_test = np.loadtxt('D:/Study/Bioinformatics/补实验/AVP/features/novel/novel_label.csv', delimiter = ',')
n_train = len(y_train)
n_test = len(y_test)


kernel_train_list = []
kernel_test_list = []
gram_train = np.zeros((n_train, n_train))
gram_test = np.zeros((n_test, n_train))

n_kernels = 5

methods_name = ['CKSAAP', 'DPC', 'AAC', 'ASDC', '188-bit']
for it in range(n_kernels):
    name = methods_name[it]
    gram_train = np.loadtxt('D:/Study/Bioinformatics/补实验/AVP/kernels/K_train_'+name+'.csv', delimiter = ',')
    gram_test = np.loadtxt('D:/Study/Bioinformatics/补实验/AVP/kernels/novel/K_novel_'+name+'.csv', delimiter = ',')
    kernel_train_list.append(gram_train)
    kernel_test_list.append(gram_test)

weight_v = [0.19784602,0.19857282,0.20743783,0.20479081,0.19135252]

for i in range(n_kernels):
    gram_train += kernel_train_list[i]*weight_v[i]
    gram_test += kernel_test_list[i]*weight_v[i]


cv = model_selection.StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)

C = 4.459440593145773

clf = svm.SVC(C = C, kernel = 'precomputed', probability = True)

print(weight_v)


clf.fit(gram_train, y_train)

y_score = clf.predict_proba(gram_test)
y_score = get_y_score(y_score)
#precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_score)

y_pred = clf.predict(gram_test)
ACC = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
sensitivity = metrics.recall_score(y_test, y_pred)
specificity = specificity_score(y_test, y_pred)
AUC = metrics.roc_auc_score(y_test, y_score)
MCC = metrics.matthews_corrcoef(y_test, y_pred)
#AUPR = get_AUPR(y_test, y_score)

print("===========================")
#print('testing:')
print(sensitivity)
print(specificity)
print(ACC)
print(MCC)
print(AUC)
#print('AUPR =', AUPR)

