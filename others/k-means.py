import numpy as np
import random
import matplotlib.pyplot as plt

X = np.random.rand(100, 2)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set(xlim=[0, 1], ylim=[0, 1], title='k-means',
       ylabel='Y-Axis', xlabel='X-Axis')


def dist(vec1, vec2):
    d = np.sqrt(np.sum(np.square(vec1 - vec2)))
    return d


def rand_cent(data, k):
    m = np.shape(data)[0] - 1
    a = range(0, m)
    index = random.sample(a, k)
    center = np.zeros((k, 2))
    for i in range(k):
        center[i] = data[index[i]]
    return center


def k_means(data, k):
    center_change = True
    m = np.shape(data)[0]
    cluster = np.zeros((1, m))
    flag = 0
    while center_change == True and flag <= 1000:
        flag = flag + 1
        center = rand_cent(data, k)
        for i in range(m):
            d = np.zeros((1, k))
            for j in range(k):
                d[0, j] = dist(data[i], center[j])
            idx = np.argmin(d[0])
            cluster[0, i] = idx
        for p in range(k):
            temp = np.zeros((m, 2))
            count = 0
            for q in range(m):
                if cluster[0, q] == p:
                    temp[count] = data[q]
                    count = count + 1
            new_x = np.average(temp[:, 0])
            new_y = np.average(temp[:, 1])
            if (abs(new_x - center[p, 0]) <= 0.001) and (abs(new_y - center[p, 1]) <= 0.001):
                center_change = False
            else:
                center_change = True
                center[p, 0] = new_x
                center[p, 1] = new_y
    mark = ['o', '+', '^', 'x', 'D', 'p', '*']
    col = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for a in range(k):
        temp = np.zeros((m, 2))
        count = 0
        for b in range(m):
            if cluster[0, b] == a:
                temp[count] = data[b]
                count = count + 1
        plt.scatter(temp[:, 0], temp[:, 1], marker=mark[a], color=col[a])


k = int(input())
k_means(X, k)
plt.show()
