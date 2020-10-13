import csv
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn import model_selection


def get_matrix(file):
    m = np.shape(file)[0]
    n = np.shape(file)[1]
    d = np.zeros((m, n-1))
    for index in range(m):
        d[index] = file[index][1:]
    return d


dataset_name = ['Antifp_Main', 'Antifp_DS1', 'Antifp_DS2']

for ds in range(2, 3):
    name_ds = dataset_name[ds]
    print('dataset:', name_ds)

    y_train = np.loadtxt('D:/Study/Bioinformatics/补实验/AFP/feature_matrix/' + name_ds + '/train_label.csv', delimiter = ',')
    y_test = np.loadtxt('D:/Study/Bioinformatics/补实验/AFP/feature_matrix/' + name_ds + '/test_label.csv', delimiter = ',')  

    n_train = len(y_train)
    n_test = len(y_test)

    train_CAT = np.zeros((n_train, 1))
    test_CAT = np.zeros((n_test, 1))

    for it in range(5):
        methods = ['188-bit', 'AAC', 'ASDC', 'CKSAAP', 'DPC']
        name = methods[it]
        print(name)

        f1 = np.loadtxt('D:/Study/Bioinformatics/补实验/AFP/feature_matrix/' + name_ds + '/' + name +'/train_' + name +'.csv', delimiter = ',', skiprows = 1)
        f3 = np.loadtxt('D:/Study/Bioinformatics/补实验/AFP/feature_matrix/' + name_ds + '/' + name +'/test_' + name +'.csv', delimiter = ',', skiprows = 1)
        
        X_train = get_matrix(f1)
        X_test = get_matrix(f3)

        train_CAT = np.column_stack((train_CAT, X_train))
        test_CAT = np.column_stack((test_CAT, X_test))
    
    train_CAT = train_CAT[:,1:]
    test_CAT = test_CAT[:,1:]


    with open('D:/Study/Bioinformatics/补实验/AFP/feature_matrix/'+name_ds+'/CAT/train_CAT.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in train_CAT:
            writer.writerow(row)
        csvfile.close()
    with open('D:/Study/Bioinformatics/补实验/AFP/feature_matrix/'+name_ds+'/CAT/test_CAT.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in test_CAT:
            writer.writerow(row)
        csvfile.close()


