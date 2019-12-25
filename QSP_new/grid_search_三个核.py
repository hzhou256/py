import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection
from imblearn.metrics import specificity_score
from sklearn.model_selection import GridSearchCV


f1 = np.loadtxt('D:/Study/Bioinformatics/QSP_new/kernel_matrix_3/KM_train_tanimoto/combine_tanimoto_train.csv', delimiter = ',')
f2 = np.loadtxt('D:/Study/Bioinformatics/QSP_new/train_label.csv', delimiter = ',')

np.set_printoptions(suppress = True)
gram = f1
y_train = f2

cv = model_selection.StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)
parameters = {'C': np.logspace(-10, 10, base = 10)}
clf = GridSearchCV(svm.SVC(kernel = 'precomputed', probability = True), parameters, n_jobs = -1, cv = cv)
clf.fit(gram, y_train)
print(clf.best_score_)
print(clf.best_params_)






