import numpy as np
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold


dataset = ['australian', 'breastw', 'diabetes', 'german', 'heart', 'ionosphere', 'sonar', 'mushroom', 'bupa', 'transfusion', 'spam']
for i in range(3, 4):
    name = dataset[i]
    print(name)
    f1 = np.loadtxt('E:/Study/Bioinformatics/UCI/' + name + '/data.csv', delimiter = ',')
    X = f1[:, 0:-1]
    y = f1[:, -1]

    cnt_0, cnt_1 = 0, 0
    for i in range(len(y)):
        if y[i] == -1:
            y[i] = 0
            cnt_0 += 1
        elif y[i] == 1:
            cnt_1 += 1

    max_val = int(min(cnt_0, cnt_1)/5*4)
    print(max_val)
    num = int((max_val - max_val%5)/5)
    print(num)