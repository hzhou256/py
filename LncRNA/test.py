import numpy as np
import My_Fuzzy_SVM
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn import metrics, preprocessing, svm
from imblearn.metrics import specificity_score
import membership


def get_feature(file):
    m = np.shape(file)[0]
    n = np.shape(file)[1]
    data = np.zeros((m, n-1))
    for index in range(m):
        data[index] = file[index][1:]
    return data

np.set_printoptions(suppress = True)
f1 = np.loadtxt('E:/Study/Bioinformatics/RNA/dataset/train.csv', delimiter = ',', skiprows = 1)
X_train = get_feature(f1)
y_train = f1[:, 0]


s = membership.SVDD_membership(X_train, y_train, g = 32, C = 0.4)
w = np.linspace(0, 1, num = 20)
print(w)
