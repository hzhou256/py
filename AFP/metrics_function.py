import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist



def gaussian(vec1, vec2, g):
    k = np.exp(-g*np.square((np.linalg.norm(vec1 - vec2))))
    return k


def cosine(X, Y):
    n = np.shape(X)[0]
    m = np.shape(Y)[0]
    K = np.ones((n, m)) - cdist(X, Y, 'cosine')
    return K

def pearson(vec1, vec2):
    X = np.vstack([vec1,vec2])
    d = np.corrcoef(X)[0][1]
    return d

def tanimoto_base(p_vec, q_vec):
    pq = np.dot(p_vec, q_vec)
    p_square = np.square(np.linalg.norm(p_vec))
    q_square = np.square(np.linalg.norm(q_vec))
    d = pq / (p_square + q_square - pq)
    return d

def tanimoto(X, Y):
    K = cdist(X, Y, tanimoto_base)
    return K