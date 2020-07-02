import collections
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, ClassifierMixin


class DVM(BaseEstimator, ClassifierMixin):
    '''
    linear DVM
    '''
    def __init__(self, beta = 0.1, lamda = 0.1, n_neighbors = 5, knn_metric = 'minkowski', Laplacian_metric = 'cosine', p = 2):
        self.beta = beta
        self.lamda = lamda
        self.n_neighbors = n_neighbors
        self.knn_metric = knn_metric
        self.Laplacian_metric = Laplacian_metric
        self.p = p

    
    def Ak_matrix(self, n_neighbors, query_index, X, X_test, y):
        '''
        return matrix: (n_features, n_neighbors*n_class)
        '''
        n_features = np.shape(X)[1]
        cnt = dict(collections.Counter(y))
        n_class = len(cnt)
        Ak = np.zeros((n_neighbors*n_class, n_features))
        i, j = 0, 0
        index_per_class = np.zeros(n_class)
        for class_label in range(n_class):
            y_index = (y == class_label)
            X_class = X[y_index]
            neigh = NearestNeighbors(n_neighbors = n_neighbors, metric = self.knn_metric, p = self.p, n_jobs = -1)
            neigh.fit(X_class)
            x = np.reshape(X_test[query_index], (1, -1))
            knn_index = neigh.kneighbors(x, return_distance = False)
            for index in knn_index[0]:
                Ak[i] = X_class[index]
                i += 1
            index_per_class[j] = i
            j += 1
        return np.transpose(Ak), index_per_class

    def Laplacian_matrix(self, Ak_matrix):
        '''
        Laplacian matrix: (n_negihbors * n_class, n_negihbors * n_class)
        '''
        Ak = np.transpose(Ak_matrix)
        W = cdist(Ak, Ak, metric = self.Laplacian_metric)
        row_sum = np.sum(W, axis = 1)
        D = np.diag(row_sum)
        return D - W

    def get_alphak(self, query_index, X, X_test, Ak, L, beta, lamda):
        I = np.identity(np.shape(Ak)[1])
        x = np.reshape(X_test[query_index], (-1, 1))
        temp = np.dot(np.transpose(Ak), Ak) + beta*I + lamda*L
        inverse = np.linalg.inv(temp)
        alphak = np.dot(inverse, np.transpose(Ak)).dot(x)
        return alphak

    def get_residue_for_one_class(self, query_index, X_test, Ak_i, alphak_i):
        x = np.reshape(X_test[query_index], (-1, 1))
        temp = x - np.dot(Ak_i, alphak_i)
        residue = np.linalg.norm(temp)
        return residue

    '''
    def fit(self, X, y):
        cnt = dict(collections.Counter(y))
        n_class = len(cnt)
        n_samples, n_features = np.shape(X)
        for query_index in range(n_samples):
            Ak = Ak_matrix(n_neighbors = self.n_neighbors, query_index = query_index, X = X, y = y)
            L = Laplacian_matrix(Ak)
            alphak = get_alphak(query_index = query_index, X = X, Ak = Ak, L = L, beta = self.beta, lamda = self.lamda)
    '''

    def predict(self, X, y, X_test):
        cnt = dict(collections.Counter(y))
        n_class = len(cnt)
        n_samples = len(X_test)
        y_predict = np.zeros(n_samples)
        for query_index in range(n_samples):
            print(query_index)
            Ak, index_per_class = self.Ak_matrix(n_neighbors = self.n_neighbors, query_index = query_index, X = X, X_test = X_test, y = y)
            L = self.Laplacian_matrix(Ak)
            alphak = self.get_alphak(query_index = query_index, X = X, X_test = X_test, Ak = Ak, L = L, beta = self.beta, lamda = self.lamda)
            residues = np.zeros(n_class)
            for class_label in range(n_class):
                Ak_i = Ak[:, int(index_per_class[class_label]-5):int(index_per_class[class_label])]
                alphak_i = alphak[int(index_per_class[class_label]-5):int(index_per_class[class_label])]
                residues[class_label] = self.get_residue_for_one_class(query_index = query_index, X_test = X_test, Ak_i = Ak_i, alphak_i = alphak_i)
            y_predict[query_index] = np.argmin(residues)
        return y_predict
    '''
    def predict_proba(self, X):

    def decision_function(self, X):
    '''