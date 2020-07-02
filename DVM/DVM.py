import collections
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


def find_knn(n_neighbors, X, y, class_label, knn_metric, p):
    '''
    return the index of kNN in X_class of each sample in X
    '''
    y_index = (y == class_label)
    X_class = X[y_index]
    neigh = NearestNeighbors(n_neighbors = n_neighbors, metric = knn_metric, p = p, n_jobs = -1)
    neigh.fit(X_class)
    knn_index_matrix = neigh.kneighbors(X, return_distance = False)
    return knn_index_matrix

def get_Ak_matrix(n_neighbors, query_index, X, y, knn_metric = 'minkowski', p = 2):
    _, n_features = np.shape(X)
    cnt = dict(collections.Counter(y))
    n_class = len(cnt)
    Ak = np.zeros((n_neighbors*n_class, n_features))
    i, j = 0, 0
    index_per_class = np.zeros(n_class)
    for class_label in range(n_class):
        y_index = (y == class_label)
        X_class = X[y_index]
        neigh = NearestNeighbors(n_neighbors = n_neighbors, metric = knn_metric, p = p, n_jobs = -1)
        neigh.fit(X_class)
        x = np.reshape(X[query_index], (1, -1))
        knn_index = neigh.kneighbors(x, return_distance = False)
        for index in knn_index[0]:
            Ak[i] = X_class[index]
            i += 1
        index_per_class[j] = i
        j += 1
    return np.transpose(Ak), index_per_class

'''
def get_Ak_matrix(n_neighbors, query_index, X, knn_index_matrix, n_class):
    _, n_features = np.shape(X)
    Ak = np.zeros((n_neighbors * n_class, n_features))
    i = 0
    for index in knn_index_matrix[query_index]:
        Ak[i] = X[index]
        i += 1
    return np.transpose(Ak)
'''

def get_Laplacian_matrix(Ak_matrix, W_metric = 'cosine'):
    '''
    Laplacian matrix: (n_negihbors * n_class, n_negihbors * n_class)
    '''
    Ak = np.transpose(Ak_matrix)
    W = cdist(Ak, Ak, metric = W_metric)
    row_sum = np.sum(W, axis = 1)
    D = np.diag(row_sum)
    return D - W

def get_alphak(query_index, X, Ak, L, beta, lamda):
    I = np.identity(np.shape(Ak)[1])
    x = np.reshape(X[query_index], (-1, 1))
    temp = np.dot(np.transpose(Ak), Ak) + beta*I + lamda*L
    inverse = np.linalg.inv(temp)
    alphak = np.dot(inverse, np.transpose(Ak)).dot(x)
    return alphak

def get_residue(query_index, X, Ak, alphak):
    x = np.reshape(X[query_index], (-1, 1))
    temp = x - np.dot(Ak, alphak)
    residue = np.linalg.norm(temp)
    return residue


f1 = np.loadtxt('E:/Study/Bioinformatics/UCI/australian/data.csv', delimiter = ',')
X = f1[:, 0:-1]
y = f1[:, -1]

for i in range(len(y)):
    if y[i] == -1:
        y[i] = 0

Ak, index_per_class = get_Ak_matrix(n_neighbors = 5, query_index = 0, X = X, y = y)
L = get_Laplacian_matrix(Ak)
alphak = get_alphak(query_index = 0, X = X, Ak = Ak, L = L, beta = 0.1, lamda = 0.1)
print(alphak)
print(index_per_class)
for class_label in range(2):
    Ak_i = Ak[:, int(index_per_class[class_label]-5):int(index_per_class[class_label])]
    alphak_i = alphak[int(index_per_class[class_label]-5):int(index_per_class[class_label])]
    print(Ak_i)
    print(alphak_i)