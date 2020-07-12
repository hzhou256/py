import collections
import numpy as np
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold


dataset = ['australian', 'breastw', 'diabetes', 'german', 'heart', 'ionosphere', 'sonar', 'mushroom', 'bupa', 'transfusion', 'spam']
for i in range(0, 1):
    name = dataset[i]
    print(name)
    f1 = np.loadtxt('E:/Study/Bioinformatics/UCI/' + name + '/data.csv', delimiter = ',')
    X = f1[:, 0:-1]
    y = f1[:, -1]

    neigh = NearestNeighbors(n_neighbors = 5, algorithm = 'kd_tree', n_jobs = -1)
    neigh.fit(X) # normal KNN
    dist, index = neigh.kneighbors(return_distance = True)
    #print(dist, index)
    rho = np.average(dist, axis = 1)
    dist = metrics.pairwise.euclidean_distances(X, squared = True)
    local_dist = np.copy(dist)
    for i in range(len(X)):
        local_dist[i] = dist[i] / rho[i]
        local_dist[:, i] = dist[:, i] / rho[i]
    W = np.exp(-local_dist)
    print(W)