from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt


X = [1,2,3,4,5]
y_ACC_Main = [0.8848,0.8964,0.8994,0.9011,0.9084]
y_MCC_Main = [0.7705,0.7937,0.7995,0.8027,0.8173]
y_ACC_DS1 = [0.8861,0.8896,0.8930,0.8870,0.8947]
y_MCC_DS1 = [0.7738,0.7801,0.7868,0.7749,0.7899]
y_ACC_DS2 = [0.9251,0.9311,0.9354,0.9354,0.9418]
y_MCC_DS2 = [0.8507,0.8628,0.8716,0.8717,0.8846]



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
    plt.text(x = np.argmax(y_ACC), y = max_ACC, s = 'ACC = ' + str(max_ACC), ha = 'left')
    plt.text(x = np.argmax(y_MCC), y = max_MCC, s = 'MCC = ' + str(max_MCC), ha = 'left')
    plt.legend(loc = 'lower right', fontsize = 9)
    plt.xlabel("Kernel number", font)
    plt.ylabel("Score", font)
    #plt.ylim((0.6, 1))
    plt.xticks([1,2,3,4,5])
    plt.title(name_ds + " Gaussian")
    plt.tight_layout()
    plt.savefig("D:\\Study\\论文\\achemso\\figure\\kernels\\" + name_ds + "_Gaussian_.png", dpi = 600)
    plt.show()