import numpy as np
import csv
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection
from sklearn.metrics import hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd
import win32api, win32con


#AMP = {'antibacterial':1,'anticancer/tumor':2,'antifungal':3,'anti-HIV':4,'antiviral':5}

f1 = np.loadtxt('D:/Study/Bioinformatics/AMP_smote/kernel_matrix_2/KM_train_cosine/combine_cosine_train.csv', delimiter = ',')
f2 = np.loadtxt('D:/Study/Bioinformatics/AMP_smote/train_label_2.csv', delimiter = ',')
n = np.shape(f2)[0]

label = np.zeros((n, 1))
for i in range(n):
    label[i][0] = f2[i]
mb = MultiLabelBinarizer()
label_mb = mb.fit_transform(label)
print(label_mb)
np.set_printoptions(suppress = True)

gram = f1
y_train = label_mb

binary_model = svm.SVC(kernel = 'precomputed', probability = True)
multi_model = OneVsRestClassifier(binary_model)
multi_model.fit(gram, y_train)

loo = model_selection.LeaveOneOut()

y_cv = model_selection.cross_val_predict(multi_model, gram, y_train, cv = loo, method = 'predict', n_jobs = -1)

for c in range(5):
    TN = 0
    TP = 0
    FN = 0
    FP = 0
    for i in range(n):
        if y_train[i][c] == 1 and y_cv[i][c] == 1:
            TP = TP + 1
        elif y_train[i][c] == 1 and y_cv[i][c] == 0:
            FN = FN + 1
        elif y_train[i][c] == 0 and y_cv[i][c] == 1:
            FP = FP + 1
        elif y_train[i][c] == 0 and y_cv[i][c] == 0:
            TN = TN + 1
    ACC = (TN + TP) / (TN + TP + FN + FP)
    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    print('class' + str(c+1) + '_ACC =', ACC)
    print('class' + str(c+1) + '_SN =', SN)
    print('class' + str(c+1) + '_SP =', SP)

win32api.MessageBox(0, "运行完毕！", "提醒", win32con.MB_OK)