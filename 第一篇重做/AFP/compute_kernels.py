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


G_list_Main = [0.25,8,1,0.25,1]
G_list_DS1 = [0.5,16,1,0.25,0.5]
G_list_DS2 = [0.25,8,0.5,0.25,1]

dataset_name = ['Antifp_Main', 'Antifp_DS1', 'Antifp_DS2']

for ds in range(2, 3):
    name_ds = dataset_name[ds]
    print('dataset:', name_ds)

    if name_ds == 'Antifp_Main':
        G_list = G_list_Main
    elif name_ds == 'Antifp_DS1':
        G_list = G_list_DS1  
    elif name_ds == 'Antifp_DS2':
        G_list = G_list_DS2 


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

        K_train = metrics.pairwise.rbf_kernel(X_train, X_train, gamma=G_list[it])
        K_test = metrics.pairwise.rbf_kernel(X_test, X_train, gamma=G_list[it])

        print(K_train)
        print(K_test)

        with open('D:/Study/Bioinformatics/补实验/AFP/kernel_matrix/'+name_ds+'/K_train_'+name+'.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in K_train:
                writer.writerow(row)
            csvfile.close()
        with open('D:/Study/Bioinformatics/补实验/AFP/kernel_matrix/'+name_ds+'/K_test_'+name+'.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in K_test:
                writer.writerow(row)
            csvfile.close()


