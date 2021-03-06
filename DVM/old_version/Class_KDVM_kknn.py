import collections
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import fractional_matrix_power
from sklearn.base import BaseEstimator, ClassifierMixin


class KDVM(BaseEstimator, ClassifierMixin):
    '''
    Kernelized DVM
    '''
    def __init__(self, beta = 0.1, lamda = 0.1, n_neighbors = 5, kernel = 'rbf', gamma = 0.5):
        self.beta = beta
        self.lamda = lamda
        self.n_neighbors = n_neighbors
        self.kernel = kernel
        self.gamma = gamma

    def kernel_matrix(self, X, Y):
        if self.kernel == 'rbf':
            K = pairwise.rbf_kernel(X, Y, gamma = self.gamma)
        elif self.kernel == 'linear':
            K = pairwise.linear_kernel(X, Y)
        
        return K



    def Laplacian_matrix(self, Ak_matrix):
        '''
        Laplacian matrix: (n_negihbors * n_class, n_negihbors * n_class)
        '''
        Ak = np.transpose(Ak_matrix)
        W = self.kernel_matrix(Ak, Ak)
        row_sum = np.sum(W, axis = 1)
        D = np.diag(row_sum)
        L_temp = D - W
        temp = fractional_matrix_power(D, -0.5)
        L = np.dot(temp, L_temp).dot(temp)
        return L


    def get_alphak(self, query_index, X_test, Ak, L, beta, lamda):
        I = np.identity(np.shape(Ak)[1])
        x = np.reshape(X_test[query_index], (1, -1))
        Ak = np.transpose(Ak)

        Gram_Ak = self.kernel_matrix(Ak, Ak)
        Gram_Ak_x = self.kernel_matrix(Ak, x)

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

        knn_index_list = []
        for class_label in range(n_class):
            y_index = (y == class_label)
            X_class = X[y_index]
            
            Gram_X_class = self.kernel_matrix(X_class, X_class)
            Gram_X_test_X_class = self.kernel_matrix(X_test, X_class)
            Gram_X_test = self.kernel_matrix(X_test, X_test)

            dist_X_class = self.get_dist_matrix(Gram_X_class, Gram_X_class, Gram_X_class)
            dist_X_test_X_class = self.get_dist_matrix(Gram_X_test, Gram_X_class, Gram_X_test_X_class)

            neigh.fit(dist_X_class)
            knn_index = neigh.kneighbors(dist_X_test_X_class, return_distance = False)
            knn_index_list += [knn_index]

        for query_index in range(n_tests):
            Ak = np.zeros((n_neighbors*n_class, n_features))
            index_per_class = np.zeros(n_class)
            i, j = 0, 0
            for class_label in range(n_class):
                y_index = (y == class_label)
                X_class = X[y_index]
                knn_index = knn_index_list[class_label][query_index]
                Ak[i:i+n_neighbors] = X_class[knn_index[:]]
                i += n_neighbors
                index_per_class[j] = i
                j += 1
            
            Ak_list[query_index] = np.transpose(Ak)
            index_list[query_index] = index_per_class
            L = self.Laplacian_matrix(Ak_list[query_index])
            alphak_list[query_index] = self.get_alphak(query_index = query_index, X_test = X_test, Ak = Ak_list[query_index], L = L, beta = self.beta, lamda = self.lamda)

        return Ak_list, index_list, alphak_list

    def get_residue(self, query_index, X_test, Ak_i, alphak_i):
        x = np.reshape(X_test[query_index], (1, -1))
        Ak_i = np.transpose(Ak_i)

        Gram_x = self.kernel_matrix(x, x)
        Gram_Aki = self.kernel_matrix(Ak_i, Ak_i)
        Gram_x_Aki = self.kernel_matrix(x, Ak_i)
        
        temp = Gram_x - 2*np.dot(Gram_x_Aki, alphak_i) + np.dot(np.transpose(alphak_i), Gram_Aki).dot(alphak_i)
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

        for query_index in range(n_tests):
            Ak, index_per_class = self.Ak_list[query_index], self.index_list[query_index]
            alphak = self.alphak_list[query_index]
            residues = np.zeros(n_class)
            for class_label in range(n_class):
                Ak_i = Ak[:, int(index_per_class[class_label]-self.n_neighbors):int(index_per_class[class_label])]
                alphak_i = alphak[int(index_per_class[class_label]-self.n_neighbors):int(index_per_class[class_label])]
                residues[class_label] = self.get_residue(query_index = query_index, X_test = X_test, Ak_i = Ak_i, alphak_i = alphak_i)
            y_predict[query_index] = np.argmin(residues)

        return y_predict

    def fit_predict(self, X, y, X_test):
        self.fit(X, y)
        y_pred = self.predict(X_test)
        return y_pred
    
    '''
    def predict_proba(self, X):

    def decision_function(self, X):
    '''
