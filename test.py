import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, KFold, train_test_split


digits = datasets.load_digits()

n_samples = len(digits.images) # images: {ndarray} of shape (1797, 8, 8) The raw image data (8*8).
data = digits.images.reshape((n_samples, -1)) 

X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)

cv = KFold(n_splits=5, shuffle=True, random_state=0)
parameters = {'estimator__C': np.logspace(-15, 5, base=2, num=21), 'estimator__gamma':np.logspace(-15, 5, base=2, num=21)}
grid = GridSearchCV(OneVsRestClassifier(SVC(kernel='rbf'), n_jobs=-1), parameters, n_jobs=-1,cv=cv, verbose=2)

grid.fit(X_train, y_train)
C = grid.best_params_['estimator__C']
gamma = grid.best_params_['estimator__gamma']


clf = OneVsRestClassifier(SVC(kernel='rbf', C=C, gamma=gamma))
clf.fit(X_train, y_train)
y_predicted = clf.predict(X_test)
ACC = metrics.accuracy_score(y_test, y_predicted)


print(ACC)
