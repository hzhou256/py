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

def get_AUPR(y_true, y_score):
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_score, pos_label = 1)
    AUPR = metrics.auc(recall, precision)
    return AUPR

font = {'size': 14}
font_legend = {'size': 10}
plt.figure(figsize = [4, 4])

dataset_name = ['Antifp_Main', 'Antifp_DS1', 'Antifp_DS2']
for ds in range(2, 3):
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
        precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_score)
        AUPR = get_AUPR(y_test, y_score)
        AUPR = round(AUPR, 4)
        plt.plot(recall, precision, label = 'Naive Bayes - AUPR: ' + str(AUPR))

        #随机森林
        clf = RandomForestClassifier(max_depth = 65, n_estimators = 125)
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)
        y_score = get_y_score(y_score)
        precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_score)
        AUPR = get_AUPR(y_test, y_score)
        AUPR = round(AUPR, 4)
        plt.plot(recall, precision, label = 'Random Forest - AUPR: ' + str(AUPR))

        #决策树
        clf = DecisionTreeClassifier(max_depth = 5, min_samples_leaf = 6)
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)
        y_score = get_y_score(y_score)
        precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_score)
        AUPR = get_AUPR(y_test, y_score)
        AUPR = round(AUPR, 4)
        plt.plot(recall, precision, label = 'Decision Tree - AUPR: ' + str(AUPR))

        #支持向量机
        clf = svm.SVC(C = 8, gamma = 3.0517578125e-05, kernel = 'rbf', probability = True)
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)
        y_score = get_y_score(y_score)
        precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_score)
        AUPR = get_AUPR(y_test, y_score)
        AUPR = round(AUPR, 4)
        plt.plot(recall, precision, label = 'SVM - AUPR: ' + str(AUPR))        

        #Our Model
        G_list = [0.25,8,0.5,0.25,1]
        weight_v = [0.20454586,0.23384408,0.22289349,0.1704432,0.16827337]

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


        clf = svm.SVC(C = 1, kernel = 'precomputed', probability = True)
        clf.fit(gram_train, y_train)

        y_score = clf.predict_proba(gram_test)
        y_score = get_y_score(y_score)
        precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_score)
        AUPR = get_AUPR(y_test, y_score)
        AUPR = round(AUPR, 4)
        plt.plot(recall, precision, label = 'Our Model - AUPR: ' + str(AUPR))  

        plt.legend(prop = font_legend)
        plt.title(name_ds, font)
        plt.xlabel('Recall', font)
        plt.ylabel('Precision', font)

        plt.tight_layout()
        plt.savefig("D:\\Study\\论文\\achemso_0825\\figure\\ROC_PR_fix\\diff_model\\PR_" + name_ds + ".png", dpi=600)
        plt.show()