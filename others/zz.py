import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from sklearn import svm
from sklearn import model_selection
from hyperopt import tpe
from hpsklearn import HyperoptEstimator, svc


def get_feature(file):
    m = np.shape(file)[0]
    n = np.shape(file)[1]
    data = np.zeros((m, n-1))
    for index in range(m):
        data[index] = file[index][1:]
    return data


dataset_name = ['Antifp_Main', 'Antifp_DS1', 'Antifp_DS2']
methods_name = ['188-bit', 'AAC', 'ASDC', 'CKSAAP', 'CTD', 'DPC']

for ds in range(1):
    name_ds = dataset_name[ds]
    print('dataset:', name_ds)
    for it in range(1):
        name = methods_name[it]

        f1 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds + '/' + name +'/train_' + name +'.csv', delimiter = ',', skiprows = 1)
        f2 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds + '/train_label.csv', delimiter = ',')
        f3 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds + '/' + name +'/test_' + name +'.csv', delimiter = ',', skiprows = 1)
        f4 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds + '/test_label.csv', delimiter = ',')

        np.set_printoptions(suppress = True)
        X_train = get_feature(f1)
        y_train = f2
        X_test = get_feature(f3)
        y_test = f4
        clf = svm.SVC(kernel = 'rbf', probability = True)
        
        if __name__ == '__main__':
            estim = HyperoptEstimator(classifier = svc('clf'), algo=tpe.suggest, max_evals=100, trial_timeout=120)
            estim.fit(X_train, y_train, n_folds=5, cv_shuffle=True, random_state=0)
            print(estim.score(X_test, y_test))
            print(estim.best_model())