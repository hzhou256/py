from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np

f1 = np.loadtxt("D:/study/Bioinformatics/QSP/300p_300n/CTD/train_CTD.csv", delimiter = ',', skiprows = 1)
m = np.shape(f1)[0]
n = np.shape(f1)[1]
data = np.zeros((m, n-1))
for index in range(m):
    data[index] = f1[index][1:]
f2 = np.loadtxt('D:/study/Bioinformatics/QSP/300p_300n/train_label.csv', delimiter = ',')

X_train = data
y_train = f2

RF = RandomForestClassifier()

parameters = {
    "n_estimators": range(1, 51),
    "criterion": ["gini", "entropy"],
    "min_samples_leaf": range(1, 11)}

clf = GridSearchCV(RF, parameters, n_jobs = -1, cv = 10)
clf.fit(X_train, y_train)
print(clf.best_score_)
print(clf.best_params_)