from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt


def get_feature(file):
    m = np.shape(file)[0]
    n = np.shape(file)[1]
    data = np.zeros((m, n-1))
    for index in range(m):
        data[index] = file[index][1:]
    return data


dataset_name = ['Antifp_Main', 'Antifp_DS1', 'Antifp_DS2']
methods_name = ['188-bit', 'AAC', 'ASDC', 'CKSAAP', 'DPC']
alphabet = ['(A)', '(B)', '(C)', '(D)', '(E)']

for ds in range(3):
    name_ds = dataset_name[ds]
    print(name_ds)
    plt.figure(figsize = [12, 8])
    for it in range(5):
        name = methods_name[it]
        print(name)
        f1 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds + '/' + name +'/train_' + name +'.csv', delimiter = ',', skiprows = 1)
        f2 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds + '/train_label.csv', delimiter = ',')

        X_train = f1
        y_train = f2
        tsne=TSNE()
        X_tsne = tsne.fit_transform(X_train)
        print(X_tsne)

        ax_train = plt.subplot(2, 3, it+1)
        n_pos = int(sum(y_train))
        n_neg = len(y_train) - n_pos
        X_tsne_pos = np.zeros((n_pos, 2))
        X_tsne_neg = np.zeros((n_neg, 2))
        j, k = 0, 0
        for i in range(len(y_train)):
            if y_train[i] == 0:
                X_tsne_neg[j] = X_tsne[i]
                j = j + 1
            else:
                X_tsne_pos[k] = X_tsne[i]
                k = k + 1

        plt.scatter(X_tsne_pos[:, 0], X_tsne_pos[:, 1], c = 'c', marker = '.', s = 8, label = 'Pos')
        plt.scatter(X_tsne_neg[:, 0], X_tsne_neg[:, 1], c = 'r', marker = '.', s = 8, label = 'Neg')
        plt.legend(loc = 'upper left')
        plt.title(alphabet[it] + '' + name, y = -0.18)
    plt.savefig("D:/论文/图表/tsne可视化/" + name_ds + ".png")
    plt.show()