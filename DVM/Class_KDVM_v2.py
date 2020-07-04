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
    def __init__(self, beta = 0.1, lamda = 0.1, n_neighbors = 5, Laplacian_metric = 'cosine', kernel = 'rbf', gamma = 0.5):
        self.beta = beta
        self.lamda = lamda
        self.n_neighbors = n_neighbors
        self.Laplacian_metric = Laplacian_metric
        self.kernel = kernel
        self.gamma = gamma

    def Laplacian_matrix(self, Ak_matrix):
        '''
        Laplacian matrix: (n_negihbors * n_class, n_negihbors * n_class)
        '''
        Ak = np.transpose(Ak_matrix)
        W = cdist(Ak, Ak, metric = self.Laplacian_metric)
        row_sum = np.sum(W, axis = 1)
        D = np.diag(row_sum)
        return D - W

    def get_alphak(self, query_index, X_test, Ak, L, beta, lamda):
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

    def get_dist_matrix(self, Gram_x, Gram_y, Gram_xy):
        m, n = np.shape(Gram_xy)
        dist = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                dist[i][j] = np.sqrt(Gram_x[i][i] + Gram_y[j][j] - 2*Gram_xy[i][j])
        return dist

    def get_matrix_all(self, n_neighbors, X, X_test, y):
        '''
        return Ak_list, index_list, L_list, alphak_list for X_test
        '''
        n_features = np.shape(X)[1]
        n_tests = np.shape(X_test)[0]
        cnt = dict(collections.Counter(y))
        n_class = len(cnt)
        neigh = NearestNeighbors(n_neighbors = n_neighbors, metric = 'precomputed', n_jobs = -1)

        Ak_list = np.zeros((n_tests, n_features, n_neighbors*n_class))
        index_list = np.zeros((n_tests, n_class))
        alphak_list = np.zeros((n_tests, n_neighbors*n_class, 1))

        widgets = ['Fit_progress: ', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA()]
        p = ProgressBar(widgets = widgets, maxval = n_tests)
        p.start()

        knn_index_list = []
        for class_label in range(n_class):
            y_index = (y == class_label)
            X_class = X[y_index]
            if self.kernel == 'rbf':
                Gram_X_class = pairwise.rbf_kernel(X_class)
                Gram_X_test_X_class = pairwise.rbf_kernel(X_test, X_class)
                Gram_X_test = pairwise.rbf_kernel(X_test)
                dist_X_class = self.get_dist_matrix(Gram_X_class, Gram_X_class, Gram_X_class)
                dist_X_test_X_class = self.get_dist_matrix(Gram_X_test, Gram_X_class, Gram_X_test_X_class)

            neigh.fit(dist_X_class)
            knn_index = neigh.kneighbors(dist_X_test_X_class, return_distance = False)
            knn_index_list.append(knn_index)

        for query_index in range(n_tests):
            Ak = np.zeros((n_neighbors*n_class, n_features))
            index_per_class = np.zeros(n_class)
            i, j = 0, 0
            for class_label in range(n_class):
                y_index = (y == class_label)
                X_class = X[y_index]
                knn_index = knn_index_list[class_label][query_index]
                for index in knn_index:
                    Ak[i] = X_class[index]
                    i += 1
                index_per_class[j] = i
                j += 1
            
            Ak_list[query_index] = np.transpose(Ak)
            index_list[query_index] = index_per_class
            L = self.Laplacian_matrix(Ak_list[query_index])
            alphak_list[query_index] = self.get_alphak(query_index = query_index, X_test = X_test, Ak = Ak_list[query_index], L = L, beta = self.beta, lamda = self.lamda)
            time.sleep(0.01)
            p.update(query_index+1)

        p.finish()
        return Ak_list, index_list, alphak_list

    def get_residue(self, query_index, X_test, Ak_i, alphak_i):
        x = np.reshape(X_test[query_index], (-1, 1))
        temp = x - np.dot(Ak_i, alphak_i)
        residue = np.linalg.norm(temp)
        return residue

    def fit(self, X, y):
        self.X = X
        self.y = y
        cnt = dict(collections.Counter(y))
        self.n_class = len(cnt)
        return self

    def predict(self, X):
        X_test = X
        self.n_tests = len(X_test)
        self.Ak_list, self.index_list, self.alphak_list = self.get_matrix_all(n_neighbors = self.n_neighbors, X = self.X, y = self.y, X_test = X_test)
        n_tests = self.n_tests
        n_class = self.n_class

        y_predict = np.zeros(n_tests)
        
        widgets = ['Predict_progress: ', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA()]
        p = ProgressBar(widgets = widgets, maxval = n_tests)
        p.start()

        for query_index in range(n_tests):
            Ak, index_per_class = self.Ak_list[query_index], self.index_list[query_index]
            alphak = self.alphak_list[query_index]
            residues = np.zeros(n_class)
            for class_label in range(n_class):
                Ak_i = Ak[:, int(index_per_class[class_label]-self.n_neighbors):int(index_per_class[class_label])]
                alphak_i = alphak[int(index_per_class[class_label]-self.n_neighbors):int(index_per_class[class_label])]
                residues[class_label] = self.get_residue(query_index = query_index, X_test = X_test, Ak_i = Ak_i, alphak_i = alphak_i)
            y_predict[query_index] = np.argmin(residues)

            time.sleep(0.01)
            p.update(query_index+1)
        p.finish()
        return y_predict

    def fit_predict(self, X, y, X_test):
        self.fit(X, y)
        y_pred = self.predict(X_test)
        return y_pred
    
    '''
    def predict_proba(self, X):

    def decision_function(self, X):
    '''
