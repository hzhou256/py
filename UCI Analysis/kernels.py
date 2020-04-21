import sys
path='D:/Program Files/libsvm_weights-3.23/python'
sys.path.append(path)
import numpy as np
import numba
import membership, My_Fuzzy_SVM
from scipy.spatial.distance import cdist
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn import svm, preprocessing, metrics
from imblearn.metrics import specificity_score


@numba.jit(nopython = True, fastmath = True) 
def gaussian(vec1, vec2, g):
    k = np.exp(-g*np.square((np.linalg.norm(vec1 - vec2))))
    return k

def Gauss(X, Y, g):
    K = cdist(X, Y, gaussian, g = g)
    return K

def Poly(X, Y, gamma, r, degree):
    temp = gamma * np.dot(X, Y) + r
    return np.power(temp, degree)

def Sigmoid(X, Y, gamma, r):
    temp = gamma * np.dot(X, Y) + r
    return np.tanh(temp)

def get_feature(file):
    m = np.shape(file)[0]
    n = np.shape(file)[1]
    data = np.zeros((m, n-1))
    for index in range(m):
        data[index] = file[index][1:]
    return data


dataset = ['ionosphere', 'german']
name = dataset[1]

f1 = np.loadtxt('E:/Study/Bioinformatics/UCI/' + name + '/X_train.csv', delimiter = ',', skiprows = 1)
X_train = get_feature(f1)
y_train = f1[:, 0]

f2 = np.loadtxt('E:/Study/Bioinformatics/UCI/' + name + '/X_test.csv', delimiter = ',', skiprows = 1)
X_test = get_feature(f2)
y_test = f2[:, 0]

scaler = preprocessing.MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

index_column_train = np.zeros((np.shape(X_train)[0], 1))
for i in range(np.shape(X_train)[0]):
    index_column_train[i] = i + 1

index_column_test = np.zeros((np.shape(X_test)[0], 1))
for j in range(np.shape(X_test)[0]):
    index_column_test[j] = j + 1

# Gaussian kernel
'''
parameters = {'C': np.logspace(-10, 10, base = 2), 'gamma': np.logspace(5, -5, base = 10)}
clf = GridSearchCV(svm.SVC(), parameters, n_jobs = -1, cv = 5, verbose = 1)
clf.fit(X_train, y_train)
gamma = clf.best_params_['gamma']
C = clf.best_params_['C']
print('Gauss')
print('C = ', C)
print('gamma = ', gamma)
print(clf.best_score_)
'''
gamma = 0.029470517025518096
K_train_gauss = Gauss(X_train, X_train, gamma)
K_test_gauss = Gauss(X_test, X_train, gamma)

# Linear kernel
K_train_linear = np.dot(X_train, X_train.T)
K_test_linear = np.dot(X_test, X_train.T)

# Polynomial kernel
gamma = 0.0167683293681101
r = 3.7275937203149416
degree = 5
K_train_poly = Poly(X_train, X_train.T, gamma, r, degree)
K_test_poly = Poly(X_test, X_train.T, gamma, r, degree)

K_train = (K_train_gauss + K_train_linear + K_train_poly) / 3
K_test = (K_test_gauss + K_test_linear + K_test_poly) / 3
K_train_SVM = np.column_stack((index_column_train, K_train))
K_test_SVM = np.column_stack((index_column_test, K_test))

nu = 0.5
W = membership.SVDD_kernel(X_train, y_train, K_train, C = nu)
#W = []
parameters = {'C': np.logspace(-10, 10, base = 2, num = 21)}
grid = GridSearchCV(My_Fuzzy_SVM.FSVM_Classifier(W = W, kernel = 'precomputed', membership = 'precomputed'), parameters, n_jobs = -1, cv = 5, verbose = 1)
grid.fit(K_train_SVM, y_train, W)
C = grid.best_params_['C']

clf = My_Fuzzy_SVM.FSVM_Classifier(W = W, C = C, membership = 'precomputed', kernel = 'precomputed')
clf.fit(K_train_SVM, y_train, W)

scorerMCC = metrics.make_scorer(metrics.matthews_corrcoef)
scorerSP = metrics.make_scorer(specificity_score)
scorerPR = metrics.make_scorer(metrics.precision_score)
scorerSE = metrics.make_scorer(metrics.recall_score)
scorer = {'ACC':'accuracy', 'recall':scorerSE, 'roc_auc': 'roc_auc', 'MCC':scorerMCC, 'SP':scorerSP}
five_fold = cross_validate(clf, K_train_SVM, y_train, cv = 5, scoring = scorer)
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

y_pred = clf.predict(K_test_SVM)
ACC = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
sensitivity = metrics.recall_score(y_test, y_pred)
specificity = specificity_score(y_test, y_pred)
AUC = metrics.roc_auc_score(y_test, clf.decision_function(K_test_SVM))
MCC = metrics.matthews_corrcoef(y_test, y_pred)

print('Testing set:')
print('SN =', sensitivity)
print('SP =', specificity)
print('ACC =', ACC)
print('MCC =', MCC)
print('AUC =', AUC)


print('C = ', C)
print('nu =', nu)
