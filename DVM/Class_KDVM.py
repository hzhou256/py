import time
import collections
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, ClassifierMixin
from progressbar import Percentage, Bar, Timer, ETA, ProgressBar


class KDVM(BaseEstimator, ClassifierMixin):
    '''
    Kernelized DVM
    '''
    def __init__(self, beta = 0.1, lamda = 0.1, n_neighbors = 5, knn_metric = 'minkowski', p = 2, Laplacian_metric = 'cosine', kernel = 'rbf', gamma = 0.5):
        self.beta = beta
        self.lamda = lamda
        self.n_neighbors = n_neighbors
        self.knn_metric = knn_metric
        self.Laplacian_metric = Laplacian_metric
        self.p = p
        self.kernel = kernel
        self.gamma = gamma
    
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
        x = np.reshape(X_test[query_index], (1, -1))
        Ak = np.transpose(Ak)
        if self.kernel == 'rbf':
            Gram_Ak = pairwise.rbf_kernel(Ak, gamma = self.gamma)
            Gram_Ak_x = pairwise.rbf_kernel(Ak, x, gamma = self.gamma)
        temp = Gram_Ak + beta*I + lamda*L
        inverse = np.linalg.inv(temp)
        alphak = np.dot(inverse, Gram_Ak_x)
        return alphak

    def get_residue_for_one_class(self, query_index, X_test, Ak_i, alphak_i):
        x = np.reshape(X_test[query_index], (-1, 1))
        temp = x - np.dot(Ak_i, alphak_i)
        residue = np.linalg.norm(temp)
        return residue

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def predict(self, X_test):
        X = self.X
        y = self.y
        cnt = dict(collections.Counter(y))
        n_class = len(cnt)
        n_samples = len(X_test)
        y_predict = np.zeros(n_samples)
        
        widgets = ['Progress: ', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA()]
        p = ProgressBar(widgets = widgets, maxval = n_samples)
        p.start()
        
        for query_index in range(n_samples):
            Ak, index_per_class = self.Ak_matrix(n_neighbors = self.n_neighbors, query_index = query_index, X = X, X_test = X_test, y = y)
            L = self.Laplacian_matrix(Ak)
            alphak = self.get_alphak(query_index = query_index, X = X, X_test = X_test, Ak = Ak, L = L, beta = self.beta, lamda = self.lamda)
            residues = np.zeros(n_class)
            for class_label in range(n_class):
                Ak_i = Ak[:, int(index_per_class[class_label]-self.n_neighbors):int(index_per_class[class_label])]
                alphak_i = alphak[int(index_per_class[class_label]-self.n_neighbors):int(index_per_class[class_label])]
                residues[class_label] = self.get_residue_for_one_class(query_index = query_index, X_test = X_test, Ak_i = Ak_i, alphak_i = alphak_i)
            y_predict[query_index] = np.argmin(residues)
            time.sleep(0.01)
            p.update(query_index+1)
        p.finish()
        return y_predict
    
    '''
    def predict_proba(self, X):

    def decision_function(self, X):
    '''
