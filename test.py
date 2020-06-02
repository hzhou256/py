import scipy.stats
import numpy as np
from sklearn import metrics


vec1 = [1, 2, 3]
vec2 = [4, 5, 6]
vec1 = np.reshape(vec1, (1, -1))
vec2 = np.reshape(vec2, (1, -1))
K = metrics.pairwise.rbf_kernel(vec1, vec2, gamma = 0.5)
K = np.double(K)
print(K)

