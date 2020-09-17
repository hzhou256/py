import numpy as np
import scipy.io as sio
from sklearn import svm, model_selection
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist

def tanimoto_base(p_vec, q_vec):
    pq = np.dot(p_vec, q_vec)
    p_square = np.square(np.linalg.norm(p_vec))
    q_square = np.square(np.linalg.norm(q_vec))
    d = pq / (p_square + q_square - pq)
    return d

def tanimoto(X, Y):
    K = cdist(X, Y, tanimoto_base)
    return K

features = ['CKSNAP', 'Kmer4', 'Kmer1234', 'RCKmer', 'DNC', 'TNC']

mat = sio.loadmat("D:\\Study\\Bioinformatics\\王浩\\data and code\\matlab\\snoRNA\\snoRNA.mat")
matrix = mat['snoRNA_' + features[0]]
label = mat['multi_label']

X = matrix
y = label

gram_train = tanimoto(X, X)
print(gram_train)


cv = KFold(n_splits = 10, shuffle = True, random_state = 0)
parameters = {'estimator__gamma': np.logspace(5, -15, base = 2, num = 21), 'estimator__C': np.logspace(-15, 10, base = 2, num = 21)}

clf_grid = OneVsRestClassifier(svm.SVC(kernel='precomputed', probability=True))
#grid = model_selection.GridSearchCV(clf_grid, parameters, n_jobs = -1, cv = cv, verbose = 2)
#grid.fit(gram_train, y)
#C = grid.best_params_['estimator__C']
#gamma = grid.best_params_['estimator__gamma']
#print('C =', C)
#print('gamma =', gamma)

C = 3.0517578125e-05
gamma = 32.0

clf = OneVsRestClassifier(svm.SVC(kernel='precomputed', C=C, gamma=gamma, probability=True))
five_fold = model_selection.cross_validate(clf, gram_train, y, cv = cv)

mean_ACC = np.mean(five_fold['test_score'])
print(mean_ACC)