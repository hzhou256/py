import numpy as np
from matplotlib import pyplot as plt


# 生成训练数据
X = 0.3 * np.random.randn(1000, 2)
X_train = np.r_[X+2, X-2]

plt.scatter(X[:,0], X[:,1])
plt.show()

