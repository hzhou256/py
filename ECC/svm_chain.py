import numpy as np
import scipy.io as sio
from ChainModel import ClassifierChain
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.model_selection import train_test_split
from sklearn import svm
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


base_svm = svm.SVC(kernel='rbf', probability=True)
chains = [ClassifierChain(base_svm, order='random', random_state=i) for i in range(10)]
for chain in chains:
    chain.fit(X_train, y_train)
y_pred_chains = np.array([chain.predict(X_test) for chain in chains])
print(y_pred_chains)