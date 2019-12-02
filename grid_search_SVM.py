from sklearn.model_selection import GridSearchCV
from sklearn import svm
import numpy as np

f1 = np.loadtxt("C:/学习/Bioinformatics/QSP/99p_99n/CKSAAP/train_CKSAAP.csv", delimiter = ',', skiprows = 1)
m = np.shape(f1)[0]
n = np.shape(f1)[1]
data = np.zeros((m, n-1))
for index in range(m):
    data[index] = f1[index][1:]
f2 = np.loadtxt('C:/学习/Bioinformatics/QSP/99p_99n/train_label.csv', delimiter = ',')

X_train = data
y_train = f2

parameters = {'kernel':['rbf'],'C':np.logspace(-30, 30, base = 2), 'gamma':np.logspace(30, -30, base = 2)}
clf = GridSearchCV(svm.SVC(), parameters, n_jobs = -1, cv = 5)
clf.fit(X_train, y_train)
print(clf.best_score_)
print(clf.best_params_)



