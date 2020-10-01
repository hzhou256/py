from sklearn.manifold import TSNE
from sklearn.metrics import pairwise
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


def get_dist_matrix(Gram_x, Gram_y, Gram_xy):
    m, n = np.shape(Gram_xy)
    dist = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            dist[i][j] = np.sqrt(Gram_x[i][i] + Gram_y[j][j] - 2*Gram_xy[i][j])
    return dist


features = ['TNC', 'Kmer1234', 'Kmer4', 'CKSNAP', 'DNC', 'RCKmer', 'NAC']
alphabet = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)']
gamma_list = [2.00000, 0.25000, 0.50000, 1.00000, 2.00000, 1.00000, 0.00006]

plt.figure(figsize = [12, 12])
for it in range(7):
    name = features[it]
    print(name)

    mat = sio.loadmat("D:\\Study\\Bioinformatics\\王浩\\data and code\\matlab\\lncRNA\\lncRNA.mat")
    matrix = mat['lncRNA_' + features[it]]
    label = mat['multi_label']

    X = matrix
    y = label[:,:2]
    #print(y)
    idx = y[:,0] != y[:,1]
    #print(idx)

    y_train = y[idx, 1] #标签
    X_train = matrix[idx, :] #只取特征文件的矩阵部分
    

    #读取特征文件，标签

    Gram = pairwise.rbf_kernel(X_train, X_train, gamma = gamma_list[it])
    X_dist = get_dist_matrix(Gram, Gram, Gram)

    tsne=TSNE(metric="precomputed")
    X_tsne = tsne.fit_transform(X_dist) #tsne降成2维
    print(X_tsne)

    ax_train = plt.subplot(3, 3, it+1) #画子图
    #将降维后的正负样本分开，方便画图（三类就分三个）
    n_pos = int(sum(y_train)) #正样本数量
    n_neg = len(y_train) - n_pos #负样本数量
    X_tsne_pos = np.zeros((n_pos, 2)) #存储正样本
    X_tsne_neg = np.zeros((n_neg, 2)) #存储负样本
    j, k = 0, 0
    for i in range(len(y_train)):
        if y_train[i] == 0:
            X_tsne_neg[j] = X_tsne[i]
            j = j + 1
        else:
            X_tsne_pos[k] = X_tsne[i]
            k = k + 1
    #正负分别画散点图
    plt.scatter(X_tsne_pos[:, 0], X_tsne_pos[:, 1], c = 'c', marker = '.', s = 8, label = 'Pos')
    plt.scatter(X_tsne_neg[:, 0], X_tsne_neg[:, 1], c = 'r', marker = '.', s = 8, label = 'Neg')
    plt.legend(loc = 'upper left')
    #在图下面画标签
    plt.title(alphabet[it] + '' + name, y = -0.18)
#plt.savefig("D:/论文/图表/tsne可视化/" + name_ds + ".png")
plt.show()
