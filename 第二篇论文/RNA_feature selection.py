from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt


X = [1,2,3,4,5,6,7]
y_AP_snoRNA = [0.81911,0.81822,0.81971,0.81972,0.81823,0.81026,0.81066]
y_AP_lncRNA = [0.75330,0.75400,0.75837,0.75915,0.76065,0.76065,0.76033]
y_AP_miRNA = [0.79240,0.79262,0.78773,0.78337,0.78065,0.77951,0.78074]

plt.figure(figsize = [5, 5])
dataset_name = ['snoRNA', 'lncRNA', 'miRNA']
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

plt.legend(loc = 'lower left', fontsize = 9)
plt.xlabel("Features", font)
plt.ylabel("Average precision", font)
#plt.ylim((0.6, 1))
plt.xticks(X)
plt.tight_layout()
plt.savefig("D:\\Study\\论文\\achemso_mlghknn\\figure\\feature_selection\\RNA.png", dpi = 600)
plt.show()