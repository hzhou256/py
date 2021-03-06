import numpy as np
from sklearn import svm
from sklearn import naive_bayes
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import model_selection, preprocessing
from imblearn.metrics import specificity_score
import matplotlib.pyplot as plt 

def get_matrix(file):
    m = np.shape(file)[0]
    n = np.shape(file)[1]
    d = np.zeros((m, n-1))
    for index in range(m):
        d[index] = file[index][1:]
    return d

def get_y_score(y_proba):
    n = np.shape(y_proba)[0]
    temp = np.zeros(n)
    for i in range(n):
        temp[i] = y_proba[i][1]
    return temp

font = {'size': 14}
font_legend = {'size': 10}
plt.figure(figsize = [4, 4])

dataset_name = ['Antifp_Main', 'Antifp_DS1', 'Antifp_DS2']
for ds in range(1):
    name_ds = dataset_name[ds]
    print(name_ds)
    methods_name = ['CAT']
    for it in range(1):
        name = methods_name[it]

        f1 = np.loadtxt('D:/Study/Bioinformatics/补实验/AFP/feature_matrix/' + name_ds + '/' + name + '/train_' + name + '.csv', delimiter = ',')
        f2 = np.loadtxt('D:/Study/Bioinformatics/补实验/AFP/feature_matrix/' + name_ds + '/train_label.csv', delimiter = ',')
        f3 = np.loadtxt('D:/Study/Bioinformatics/补实验/AFP/feature_matrix/' + name_ds + '/' + name + '/test_' + name + '.csv', delimiter = ',')
        f4 = np.loadtxt('D:/Study/Bioinformatics/补实验/AFP/feature_matrix/' + name_ds + '/test_label.csv', delimiter = ',')        

        X_train = f1
        X_test = f3
        y_train = f2
        y_test = f4

        cv = model_selection.StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
        #朴素贝叶斯
        clf = naive_bayes.GaussianNB()
        clf.fit(X_train, y_train)

        y_score = clf.predict_proba(X_test)
        y_score = get_y_score(y_score)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
        plt.plot(fpr, tpr, label = 'Naive Bayes - AUC: 0.8404')

        #随机森林
        clf = RandomForestClassifier(max_depth = 69, n_estimators = 110)
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)
        y_score = get_y_score(y_score)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
        plt.plot(fpr, tpr, label = 'Random Forest - AUC: 0.9603')

        #决策树
        clf = DecisionTreeClassifier(max_depth = 5, min_samples_leaf = 8)
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)
        y_score = get_y_score(y_score)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
        plt.plot(fpr, tpr, label = 'Decision Tree - AUC: 0.8368')

        #支持向量机
        clf = svm.SVC(C = 16, gamma = 3.0517578125e-05, kernel = 'rbf', probability = True)
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)
        y_score = get_y_score(y_score)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
        plt.plot(fpr, tpr, label = 'SVM - AUC: 0.9406')        

        #Our Model
        G_list = [0.25,8,1,0.25,1]
        weight_v = [0.20779853,0.21165743,0.20338872,0.18990633,0.18724899]

        n_train = len(y_train)
        n_test = len(y_test)
        n_kernels = 5

        kernel_train_list = []
        kernel_test_list = []
        gram_train = np.zeros((n_train, n_train))
        gram_test = np.zeros((n_test, n_train))
        for it in range(5):
            methods = ['188-bit', 'AAC', 'ASDC', 'CKSAAP', 'DPC']
            name = methods[it]
            print(name)
            f1 = np.loadtxt('D:/Study/Bioinformatics/补实验/AFP/feature_matrix/' + name_ds + '/' + name +'/train_' + name +'.csv', delimiter = ',', skiprows = 1)
            f3 = np.loadtxt('D:/Study/Bioinformatics/补实验/AFP/feature_matrix/' + name_ds + '/' + name +'/test_' + name +'.csv', delimiter = ',', skiprows = 1)
            
            X_train = get_matrix(f1)
            X_test = get_matrix(f3)

            scaler = preprocessing.MinMaxScaler(feature_range = (0, 1)).fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            gram_train = metrics.pairwise.rbf_kernel(X_train, X_train, gamma = G_list[it])
            gram_test = metrics.pairwise.rbf_kernel(X_test, X_train, gamma = G_list[it])
            kernel_train_list.append(gram_train)
            kernel_test_list.append(gram_test)
        
        for i in range(n_kernels):
            gram_train += kernel_train_list[i]*weight_v[i]
            gram_test += kernel_test_list[i]*weight_v[i]


        clf = svm.SVC(C = 4, kernel = 'precomputed', probability = True)
        clf.fit(gram_train, y_train)

        y_score = clf.predict_proba(gram_test)
        y_score = get_y_score(y_score)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
        plt.plot(fpr, tpr, label = 'Our Model - AUC: 0.9688')  

        plt.legend(prop = font_legend)
        plt.title(name_ds, font)
        plt.xlabel('False positive rate', font)
        plt.ylabel('True positive rate', font)

        plt.tight_layout()
        plt.savefig("D:\\Study\\论文\\tcbb-AFP\\figure\\ROC_PR_fix\\diff_model\\ROC_" + name_ds + ".png", dpi=600)
        plt.show()