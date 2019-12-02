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

f1 = np.loadtxt('D:/Study/Bioinformatics/AMP_smote/kernel_matrix_3/KM_train_tanimoto/combine_tanimoto_train.csv', delimiter = ',')
f2 = np.loadtxt('D:/Study/Bioinformatics/AMP_smote/multiple_label_3.csv', delimiter = ',')
n = np.shape(f2)[0]

label = f2
print(label)
np.set_printoptions(suppress = True)

gram = f1
y_train = label

binary_model = svm.SVC(kernel = 'precomputed', probability = True)
multi_model = OneVsRestClassifier(binary_model)
multi_model.fit(gram, y_train)

loo = model_selection.LeaveOneOut()
y_cv = model_selection.cross_val_predict(multi_model, gram, y_train, cv = loo, method = 'predict', n_jobs = -1)

with open('D:/Study/Bioinformatics/AMP_smote/kernel_matrix_3/KM_train_tanimoto/y_predict.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in y_cv:
        writer.writerow(row)
    csvfile.close()

win32api.MessageBox(0, "运行完毕！", "提醒", win32con.MB_OK)