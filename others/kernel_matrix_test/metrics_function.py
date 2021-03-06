import numpy as np
from scipy.spatial.distance import pdist


def euclidean(vec1, vec2):
    d = np.linalg.norm(vec1 - vec2)
    return d

def manhattan(vec1, vec2):
    d = np.linalg.norm(vec1 - vec2, ord = 1)
    return d

def chebyshev(vec1, vec2):
    d = np.linalg.norm(vec1 - vec2, ord = np.inf)
    return d

def gaussian(vec1, vec2, g):
    k = np.exp(-g*np.square((np.linalg.norm(vec1 - vec2))))
    return k

def cosine(vec1, vec2):
    X = np.vstack([vec1,vec2])
    d = float(1 - pdist(X, 'cosine'))
    return d

def pearson(vec1, vec2):
    X = np.vstack([vec1,vec2])
    d = np.corrcoef(X)[0][1]
    return d

def tanimoto(p_vec, q_vec):
    pq = np.dot(p_vec, q_vec)
    p_square = np.square(np.linalg.norm(p_vec))
    q_square = np.square(np.linalg.norm(q_vec))
    d = pq / (p_square + q_square - pq)
    return d
