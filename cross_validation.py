import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection


f1 = np.loadtxt('C:/学习/Bioinformatics/QSP/300p_300n/ASDC/train_ASDC.csv', delimiter = ',', skiprows = 1)
f2 = np.loadtxt('C:/学习/Bioinformatics/QSP/300p_300n/train_label.csv', delimiter = ',')
f3 = np.loadtxt('C:/学习/Bioinformatics/QSP/300p_300n/ASDC/test_ASDC.csv', delimiter = ',', skiprows = 1)
f4 = np.loadtxt('C:/学习/Bioinformatics/QSP/300p_300n/test_label.csv', delimiter = ',')

def get_matrix(file):
    m = np.shape(file)[0]
    n = np.shape(file)[1]
    d = np.zeros((m, n-1))
    for index in range(m):
        d[index] = file[index][1:]
    return d

np.set_printoptions(suppress = True)
X_train = get_matrix(f1)
y_train = f2
X_test = get_matrix(f3)
y_test = f4

clf = svm.SVC(C = 1.5286359852052918, gamma = 8.346807447736152, kernel = 'rbf', probability = True)

scores = model_selection.cross_val_score(clf, X_train, y_train, cv = 10)

mean = np.mean(scores)
print(scores)
print(mean)