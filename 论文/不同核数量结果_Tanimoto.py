from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt


X = [1,2,3,4,5]
y_ACC_Main = [0.8938,0.9101,0.9067,0.9063,0.9067]
y_MCC_Main = [0.7879,0.8206,0.8135,0.8127,0.8136]
y_ACC_DS1 = [0.8955,0.9037,0.9058,0.9041,0.9024]
y_MCC_DS1 = [0.7914,0.8078,0.8121,0.8086,0.8053]
y_ACC_DS2 = [0.9384,0.9456,0.9431,0.9422,0.9422]
y_MCC_DS2 = [0.8770,0.8914,0.8864,0.8846,0.8847]


dataset_name = ['Antifp_Main', 'Antifp_DS1', 'Antifp_DS2']
for ds in range(3):
    name_ds = dataset_name[ds]

    font = {'size': 14}
    plt.figure(figsize = [5, 4])

    print('dataset:', name_ds)
    if ds == 0:
        y_ACC = y_ACC_Main
        y_MCC = y_MCC_Main
    elif ds == 1:
        y_ACC = y_ACC_DS1
        y_MCC = y_MCC_DS1
    else:
        y_ACC = y_ACC_DS2
        y_MCC = y_MCC_DS2     
    plt.scatter(X, y_ACC, c = 'c', marker = '.')
    plt.plot(X, y_ACC, c = 'c', label = 'ACC')
    plt.scatter(X, y_MCC, c = 'r', marker = '.')
    plt.plot(X, y_MCC, c = 'r', label = 'MCC')
    max_ACC = np.max(y_ACC)
    max_MCC = np.max(y_MCC)
    plt.text(x = np.argmax(y_ACC)+1, y = max_ACC, s = 'ACC = ' + str(max_ACC))
    plt.text(x = np.argmax(y_MCC)+1, y = max_MCC, s = 'MCC = ' + str(max_MCC))
    plt.legend(loc = 'lower right', fontsize = 9)
    plt.xlabel("Kernel number", font)
    plt.ylabel("Score", font)
    #plt.ylim((0.6, 1))
    plt.xticks([1,2,3,4,5])
    plt.title(name_ds + " Tanimoto", font)
    plt.tight_layout()
    plt.savefig("D:\\Study\\论文\\achemso\\figure\\kernels\\" + name_ds + "_Tanimoto_.png", dpi = 600)
    plt.show()