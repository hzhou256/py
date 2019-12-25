import numpy as np
import csv
import metrics_function
from numba import jit


np.set_printoptions(suppress = True)

G_list_Main = [0.0000284792011797527, 76.9918635706838, 170.80198612881, 12.804430034573, 60.6324127241912]
G_list_DS1 = [0.0000361594727090481, 82.5454327318349, 144.873841101274, 15.1053183447585, 35.778925406486]
G_list_DS2 = [0.0000786446012462563, 91.4364494328818, 160.34375429646, 9.71484730129853, 47.646239357237]

dataset_name = ['Antifp_Main', 'Antifp_DS1', 'Antifp_DS2']
methods_name = ['188-bit', 'AAC', 'ASDC', 'CKSAAP', 'DPC']

for ds in range(1, 3):
    name_ds = dataset_name[ds]
    print(name_ds)

    if name_ds == 'Antifp_Main':
        G_list = G_list_Main
    elif name_ds == 'Antifp_DS1':
        G_list = G_list_DS1  
    elif name_ds == 'Antifp_DS2':
        G_list = G_list_DS2

    for it in range(5):
        name = methods_name[it]
        print(name)
        f1 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds +'/' + name + '/test_' + name + '.csv', delimiter = ',', skiprows = 1)
        m = np.shape(f1)[0]
        n = np.shape(f1)[1]
        X_test = np.zeros((m, n-1))
        for index in range(m):
            X_test[index] = f1[index][1:]

        f2 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds +'/' + name + '/train_' + name + '.csv', delimiter = ',', skiprows = 1)
        p = np.shape(f2)[0]
        q = np.shape(f2)[1]
        X_train = np.zeros((p, q-1))
        for index in range(p):
            X_train[index] = f2[index][1:]

        @jit
        def get_gs_kernel(X_test, X_train, m, p, gamma):
            K = np.zeros((m, p))
            for i in range(m):
                for j in range(p):
                    K[i][j] = metrics_function.gaussian(X_test[i], X_train[j], gamma)
            return K
        K1 = get_gs_kernel(X_test, X_train, m, p, G_list[it])
        print(K1)
        with open('D:/Study/Bioinformatics/AFP/kernel_matrix/' + name_ds +'/KM_test_gaussian/KM_gaussian_' + name + '_test.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in K1:
                writer.writerow(row)
            csvfile.close()

