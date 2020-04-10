import numpy as np
import My_Fuzzy_SVM
from sklearn.model_selection import GridSearchCV
from sklearn import metrics, preprocessing


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

parameters = {'C': np.logspace(-10, 10, base = 2, num = 25), 'gamma': np.logspace(5, -5, base = 10, num = 25)}
grid = GridSearchCV(My_Fuzzy_SVM.FSVM_Classifier(membership = 'SVDD'), parameters, n_jobs = -1, cv = 5, verbose = 1)
grid.fit(X_train, y_train)
gamma = grid.best_params_['gamma']
C = grid.best_params_['C']


clf = My_Fuzzy_SVM.FSVM_Classifier(C = C, gamma = gamma, membership = 'SVDD')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
ACC = metrics.accuracy_score(y_test, y_pred)
print('CV_ACC = ', grid.best_score_)
print('ACC = ', ACC)
print('C = ', C)
print('g = ', gamma)
