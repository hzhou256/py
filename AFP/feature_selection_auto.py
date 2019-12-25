import numpy as np
import pandas as pd
import metrics_function
from sklearn import metrics
from sklearn import svm
from sklearn import model_selection
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier

def get_matrix(file):
    m = np.shape(file)[0]
    n = np.shape(file)[1]
    d = np.zeros((m, n-1))
    for index in range(m):
        d[index] = file[index][1:]
    return d

dataset_name = ['Antifp_Main', 'Antifp_DS1', 'Antifp_DS2']
for ds in range(1):
    name_ds = dataset_name[ds]

    y_train = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds +'/train_label.csv', delimiter = ',')

    methods_name = ['188-bit', 'ASDC', 'CKSAAP', 'CTD']
    for it in range(1,2):
        name = methods_name[it]
        f1 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds +'/' + name +'/train_' + name +'.csv', delimiter = ',', skiprows = 1)

        f2 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds +'/' + name +'/test_' + name +'.csv', delimiter = ',', skiprows = 1)

        X_train = get_matrix(f1)
        X_test = get_matrix(f2)
        print(np.shape(X_train))

        model = ExtraTreesClassifier(n_estimators = 100, random_state = 0)
        model.fit(X_train, y_train)
        selector = SelectFromModel(model, prefit = True)

        X_train_selected = selector.transform(X_train)
        print(np.shape(X_train_selected))
        X_test_selected = selector.transform(X_test)
        print(np.shape(X_test_selected))

        train = pd.DataFrame(X_train_selected)
        pd.DataFrame.to_csv(train, 'D:/Study/Bioinformatics/AFP/feature_matrix_auto/' + name_ds +'/' + name +'/train_' + name +'.csv', header = True)
        test = pd.DataFrame(X_test_selected)
        pd.DataFrame.to_csv(test, 'D:/Study/Bioinformatics/AFP/feature_matrix_auto/' + name_ds +'/' + name +'/test_' + name +'.csv', header = True)
