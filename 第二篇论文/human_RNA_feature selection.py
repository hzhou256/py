from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt


X = [1,2,3,4,5,6,7]
y_AP_snoRNA = [0.83227,0.82807,0.83558,0.83255,0.83301,0.83213,0.83082]
y_AP_lncRNA = [0.75416,0.75593,0.75643,0.75706,0.75640,0.75960,0.75832]
y_AP_miRNA = [0.77707,0.79447,0.79804,0.79857,0.79968,0.79720,0.79525]

plt.figure(figsize = [5, 5])
dataset_name = ['H_snoRNA', 'H_lncRNA', 'H_miRNA']
for ds in range(3):
    name_ds = dataset_name[ds]

    font = {'size': 12}
    #plt.figure(figsize = [5, 4])

    print('dataset:', name_ds)
    if ds == 0:
        y_AP = y_AP_snoRNA
    elif ds == 1:
        y_AP = y_AP_lncRNA
    else:
        y_AP = y_AP_miRNA    
    plt.scatter(X, y_AP, marker = '.')
    plt.plot(X, y_AP, label = name_ds)
    max_AP = np.max(y_AP)
    plt.text(x = np.argmax(y_AP)+1, y = max_AP, s = '['+str(round(np.argmax(y_AP)+1,4))+', '+str(round(max_AP,4))+']')

plt.legend(loc = 'lower left',fontsize = 9)
plt.xlabel("Features", font)
plt.ylabel("Average precision", font)
#plt.ylim((0.6, 1))
plt.xticks(X)
plt.tight_layout()
plt.savefig("D:\\Study\\论文\\achemso_mlghknn\\figure\\feature_selection\\human_RNA.png", dpi = 600)
plt.show()