import numpy as np
import collections

A1 = np.empty((3,3))
A2 = np.identity(5)
A3 = np.zeros(3)
A4 = np.zeros(3)
Alist = []
Alist.append(A2)
Alist.append(A1)
print(Alist[0][3])