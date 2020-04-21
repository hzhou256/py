import numpy as np
import My_Fuzzy_SVM
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn import metrics, preprocessing, svm
from imblearn.metrics import specificity_score
import membership
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import collections

delta = 0.001

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

tsne=TSNE()
X_tsne = tsne.fit_transform(X_train)

cnt = dict(collections.Counter(y_train))
n_pos = cnt[1]
n_neg = cnt[2]
X_tsne_pos = np.zeros((n_pos, 2)) 
X_tsne_neg = np.zeros((n_neg, 2)) 
j, k = 0, 0
for i in range(len(y_train)):
    if y_train[i] == 2:
        X_tsne_neg[j] = X_tsne[i]
        j = j + 1
    else:
        X_tsne_pos[k] = X_tsne[i]
        k = k + 1
#正负分别画散点图
plt.scatter(X_tsne_pos[:, 0], X_tsne_pos[:, 1], c = 'c', marker = '.', s = 8, label = 'Pos')
plt.scatter(X_tsne_neg[:, 0], X_tsne_neg[:, 1], c = 'r', marker = '.', s = 8, label = 'Neg')
plt.legend(loc = 'upper left')
plt.show()