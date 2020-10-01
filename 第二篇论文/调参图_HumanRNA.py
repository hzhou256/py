import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


datasets = ['human_snoRNA','human_lncRNA','human_miRNA']
names = ['human snoRNA','human lncRNA','human miRNA']

font = {'size': 12}
font_text = {'size': 10}
plt.figure(figsize = [5, 5])
plt.ylim(0.68, 0.85)

colors = [(30/255,120/255,180/255), (50/255,160/255,50/255), (255/255,127/255,14/255)]
for i in range(0, 3):
    name = datasets[i]
    res = np.loadtxt("D:\\Study\\Bioinformatics\\王浩\\data and code\\matlab\\result\\调参结果\\"+name+".txt", delimiter=",")
    
    k = res[:,0]
    AP = res[:,1]

    plt.plot(k, AP, label = names[i], c=colors[i])
    max_AP = np.max(AP)
    plt.text(x = int(k[np.argmax(AP)]-20), y = max_AP, s = '['+str(int(k[np.argmax(AP)]))+', '+str(round(max_AP,4))+']')
    
    x = [k[np.argmax(AP)], k[np.argmax(AP)]]
    y = [0.68, max_AP]
    plt.plot(x, y, c=colors[i])
    

plt.legend(loc = 'lower right', fontsize = 9)
plt.xlabel("k", font)
plt.ylabel("Average precision", font)
plt.tight_layout()
plt.savefig("D:\\Study\\论文\\achemso_mlghknn\\figure\\parameter_tune\\humanRNA.png", dpi = 600)
plt.show()