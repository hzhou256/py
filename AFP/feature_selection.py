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
for ds in range(3):
    name_ds = dataset_name[ds]

    y_train = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds +'/train_label.csv', delimiter = ',')

    methods_name = ['188-bit', 'CTD']
    for it in range(2):
        name = methods_name[it]
        f1 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds +'/' + name +'/train_' + name +'.csv', delimiter = ',', skiprows = 1)

        f2 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds +'/' + name +'/test_' + name +'.csv', delimiter = ',', skiprows = 1)

        X_train = get_matrix(f1)
        X_test = get_matrix(f2)
        print(np.shape(X_train))

        model = ExtraTreesClassifier(n_estimators = 100, random_state = 0)
        model.fit(X_train, y_train)
        thresholds = np.sort(model.feature_importances_)
        print(thresholds)

        best_ACC = 0
        best_train = X_train
        best_test = X_test

        for thres in thresholds:
            print('threshold =', thres)
            selector = SelectFromModel(model, threshold = thres, prefit = True)

            X_train_selected = selector.transform(X_train)
            print(np.shape(X_train_selected))
            X_test_selected = selector.transform(X_test)
            print(np.shape(X_test_selected))

            gram_train = metrics_function.cosine(X_train_selected, X_train_selected)
            
            clf = svm.SVC(kernel = 'precomputed', probability = False)
            try:
                clf.fit(gram_train, y_train)
            except ValueError as e:
                print ("ValueError Details : " + str(e))
                continue
            cv = model_selection.StratifiedKFold(n_splits = 5, shuffle = False)
            five_fold = model_selection.cross_validate(clf, gram_train, y_train, cv = cv, scoring = 'accuracy', n_jobs = -1)
            ACC = np.mean(five_fold['test_score'])
            print('ACC =', ACC)
            if ACC > best_ACC:
                best_ACC = ACC
                best_train = X_train_selected
                best_test = X_test_selected
        print('best_ACC =', best_ACC)
        train = pd.DataFrame(best_train)
        pd.DataFrame.to_csv(train, 'D:/Study/Bioinformatics/AFP/feature_matrix_selected/' + name_ds +'/' + name +'/train_' + name +'.csv', header = True)
        test = pd.DataFrame(best_test)
        pd.DataFrame.to_csv(test, 'D:/Study/Bioinformatics/AFP/feature_matrix_selected/' + name_ds +'/' + name +'/test_' + name +'.csv', header = True)
