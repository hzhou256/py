import csv
import optunity
import optunity.metrics
# comment this line if you are running the notebook
from openpyxl import load_workbook
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
import pandas as pd
import metrics_function
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection
import matplotlib.pyplot as plt


def get_matrix(file):
    m = np.shape(file)[0]
    n = np.shape(file)[1]
    d = np.zeros((m, n-1))
    for index in range(m):
        d[index] = file[index][1:]
    return d

def get_label(file):
    m = np.shape(file)[0]
    label = np.zeros(m)
    for index in range(m):
        label[index] = file[index][0]
    return label


#导入数据
f1 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/Antifp_Main/ASDC/train_ASDC.csv', delimiter = ',', skiprows = 1)
y_train = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/Antifp_Main/train_label.csv', delimiter = ',')

sample = get_matrix(f1)
label = y_train
print(sample)
print(label)
#we will make the cross-validation decorator once, so we can reuse it later for the other tuning task
# by reusing the decorator, we get the same folds etc.

cv_decorator = optunity.cross_validated(x=sample, y=label, num_folds=5)

def svr_rforest_tuned_acc(x_train, y_train, x_test, y_test, n_estimators, max_depth,min_samples_leaf,
                        min_samples_split):
    rf = RandomForestClassifier(n_estimators=int(n_estimators),max_features='log2',
                                max_depth=int(max_depth),min_samples_leaf=int(min_samples_leaf),
                                min_samples_split=int(min_samples_split), n_jobs=-1).fit(x_train,y_train)
    y_pre = rf.predict(x_test)
    #pcc = round(np.corrcoef(y_pre, y_test)[0][1], 5)
    acc = optunity.metrics.accuracy(y_pre, y_test)
    # auc = optunity.metrics.roc_auc(y_test, decision_values)
    return acc
    #auc = optunity.metrics.roc_auc(y_test, decision_values)
    #print(pcc_test)
    #return optunity.metrics.mse(y_test, y_pre)

svr_rforest_tuned_acc = cv_decorator(svr_rforest_tuned_acc)
# this is equivalent to the more common syntax below
# @optunity.cross_validated(x=data, y=labels, num_folds=5)
# def svm_rbf_tuned_auroc...max_features=['square', 'log'],

optimal_rbf_pars, info, _ = optunity.maximize(svr_rforest_tuned_acc, num_evals=200, n_estimators=[1, 500],
                                             max_depth=[1,100], min_samples_leaf=[1, 20],
                                             min_samples_split=[2, 20])
# when running this outside of IPython we can parallelize via optunity.pmap
# optimal_rbf_pars, _, _ = optunity.maximize(svm_rbf_tuned_auroc, 150, C=[0, 10], gamma=[0, 0.1], pmap=optunity.pmap)

print("Optimal parameters: " + str(optimal_rbf_pars))
print("ACC of tuned rf with hyper parameters %1.5f" % info.optimum)
#regressor = SVR(kernel='rbf', gamma=10 ** optimal_rbf_pars['logGamma'], C = optimal_rbf_pars['C'])
#kernel = KF.gaussianKernel1(X_test.T, X_train.T, 10 ** optimal_rbf_pars['logGamma'])
#regressor.fit(X_train,Y_train)
#y_train = regressor.predict(X_train)
#y_predict = regressor.predict(X_test)
#X_train, X_test, y_train, y_test = train_test_split(sample, label, test_size=0.2, random_state=42)
rf1 = RandomForestClassifier(n_estimators=int(optimal_rbf_pars['n_estimators']), max_features='sqrt',
                           max_depth=int(optimal_rbf_pars['max_depth']), min_samples_leaf=
                           int(optimal_rbf_pars['min_samples_leaf']), min_samples_split=
                           int(optimal_rbf_pars['min_samples_split']), n_jobs=-1).fit(sample, label)
#y1 = rf1.predict(X_test)

#print(rf1.feature_importances_)

#dict['School'] = "RUNOOB"
dict={}
for index, feature_import in enumerate(rf1.feature_importances_):
    dict[index] = feature_import
#print(dict)
order_dict = sorted(dict.items(), key=lambda x : x[1], reverse=True)
print(order_dict)

best_ACC = 0
best_feature = np.zeros(1)
rows = np.shape(sample)[0]
feature = np.zeros((rows, 2))
x, y = order_dict[0]
for p in range(rows):
    feature[p][0] = sample[p][x]
x, y = order_dict[1]
for p in range(rows):
    feature[p][1] = sample[p][x]

print(feature)
#ax = []
#ay = []
#plt.ion()

for k, v in order_dict[2:]:
    temp = np.zeros(rows)
    for q in range(rows):
        temp[q] = sample[q][k]
    feature = np.c_[feature, temp]
    print(feature)
    gram = np.zeros((rows, rows))
    for i in range(rows):
        for j in range(rows):
            gram[i][j] = round(metrics_function.cosine(feature[i], feature[j]), 6)
    print(gram)
    #G = pd.DataFrame(gram)
    #pd.DataFrame.to_csv(G, 'D:/Study/Bioinformatics/AFP/feature_matrix/Antifp_Main/188-bit/gram.csv')
    clf = svm.SVC(kernel = 'precomputed', probability = False)
    clf.fit(gram, y_train)

    cv = model_selection.StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
    five_fold = model_selection.cross_validate(clf, gram, label, cv = cv, scoring = 'accuracy', n_jobs = -1)
    ACC = np.mean(five_fold['test_score'])
    print('ACC =', ACC)
    if ACC > best_ACC:
        best_ACC = ACC
        best_feature = np.copy(feature)
    #ax.append(k)
    #ay.append(ACC)
    #plt.clf()
    #plt.plot(ax,ay)
    #plt.pause(0.1)
    #plt.ioff()
print(best_feature)
print('best_ACC =', best_ACC)

DF = pd.DataFrame(best_feature)
pd.DataFrame.to_csv(DF, 'D:/Study/Bioinformatics/AFP/feature_matrix/Antifp_Main/ASDC/train_ASDC_selected.csv')