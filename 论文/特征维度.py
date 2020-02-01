import xlrd
import numpy as np
import matplotlib.pyplot as plt


index = [0, 1, 2, 3, 4]
feature_name = ['188-bit', 'AAC', 'ASDC', 'CKSAAP', 'DPC']
dim = [188, 20, 400, 1200, 400]
plt.barh(index, dim)
plt.yticks(index, feature_name)
plt.xlabel("Feature Dimension")
plt.show()
