import numpy as np
from sklearn import svm
from sklearn import naive_bayes
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import model_selection
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

        f1 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds + '/' + name + '/train_' + name + '.csv', delimiter = ',', skiprows = 1)
        f2 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds + '/train_label.csv', delimiter = ',')
        f3 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds + '/' + name + '/test_' + name + '.csv', delimiter = ',', skiprows = 1)
        f4 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds + '/test_label.csv', delimiter = ',')        

        X_train = get_matrix(f1)
        X_test = get_matrix(f3)
        y_train = f2
        y_test = f4

        cv = model_selection.StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
        #朴素贝叶斯
        clf = naive_bayes.GaussianNB()
        clf.fit(X_train, y_train)

        y_score = clf.predict_proba(X_test)
        y_score = get_y_score(y_score)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
        plt.plot(fpr, tpr, label = 'Naive Bayes - AUC: 0.8431')

        #随机森林
        clf = RandomForestClassifier(max_depth = 92, n_estimators = 132)
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)
        y_score = get_y_score(y_score)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
        plt.plot(fpr, tpr, label = 'Random Forest - AUC: 0.9586')

        #决策树
        clf = DecisionTreeClassifier(max_depth = 6, min_samples_leaf = 9)
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)
        y_score = get_y_score(y_score)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
        plt.plot(fpr, tpr, label = 'Decision Tree - AUC: 0.8464')

        #支持向量机
        clf = svm.SVC(C = 27.087, gamma = 7.34, kernel = 'rbf', probability = True)
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)
        y_score = get_y_score(y_score)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
        plt.plot(fpr, tpr, label = 'SVM - AUC: 0.9588')        

        #Our Model
        f1 = np.loadtxt('D:/Study/Bioinformatics/AFP/kernel_matrix_2/' + name_ds + '/KM_train_tanimoto/combine_tanimoto_HSIC_train.csv', delimiter = ',')
        f2 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds + '/train_label.csv', delimiter = ',')
        f3 = np.loadtxt('D:/Study/Bioinformatics/AFP/kernel_matrix_2/' + name_ds + '/KM_test_tanimoto/combine_tanimoto_HSIC_test.csv', delimiter = ',')
        f4 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds + '/test_label.csv', delimiter = ',')

        gram = f1
        y_train = f2
        gram_test = f3
        y_test = f4

        clf = svm.SVC(C = 3.237, kernel = 'precomputed', probability = True)
        clf.fit(gram, y_train)

        y_score = clf.predict_proba(gram_test)
        y_score = get_y_score(y_score)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
        plt.plot(fpr, tpr, label = 'Our Model - AUC: 0.9695')  

        plt.legend(prop = font_legend)
        plt.title(name_ds, font)
        plt.xlabel('False positive rate', font)
        plt.ylabel('True positive rate', font)

        plt.tight_layout()
        plt.savefig("D:\\Study\\论文\\achemso\\figure\\ROC_PR_fix\\diff_model\\ROC_" + name_ds + ".png", dpi=600)
        plt.show()