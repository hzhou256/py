import numpy as np
import csv
import metrics_function

methods_name = ['188-bit', 'AAC', 'ASDC', 'CKSAAP', 'CTD']
for it in range(5):
    name = methods_name[it]
    print(name + ':')
    for k in range(10):
        file = np.loadtxt("C:/学习/Bioinformatics/QSP/200p_200n/10_fold/" + name + "/test/test_" + name + "_" + str(k) + ".csv", delimiter = ',')
        m = np.shape(file)[0]
        n = np.shape(file)[1]
        X_test = np.zeros((m, n-1))
        for index in range(m):
            X_test[index] = file[index][1:]

        file1 = np.loadtxt("C:/学习/Bioinformatics/QSP/200p_200n/10_fold/" + name + "/train/train_" + name + "_" + str(k) + ".csv", delimiter = ',')
        p = np.shape(file1)[0]
        q = np.shape(file1)[1]
        X_train = np.zeros((p, q-1))
        for index in range(p):
            X_train[index] = file1[index][1:]

        K1 = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                K1[i][j] = round(metrics_function.cosine(X_train[i], X_train[j]), 6)
        print(K1)
        with open('C:/学习/Bioinformatics/QSP/200p_200n/10_fold/' + name + '/km_train/KM_cosine_' + name + '_train_' + str(k) + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in K1:
                writer.writerow(row)
            csvfile.close()

        K2 = np.zeros((m, p))
        for i in range(m):
            for j in range(p):
                K2[i][j] = round(metrics_function.cosine(X_test[i], X_train[j]), 6)
        print(K2)
        with open('C:/学习/Bioinformatics/QSP/200p_200n/10_fold/' + name + '/km_test/KM_cosine_' + name + '_test_' + str(k) + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in K2:
                writer.writerow(row)
            csvfile.close()

        K3 = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                K3[i][j] = round(metrics_function.tanimoto(X_train[i], X_train[j]), 6)
        print(K3)
        with open('C:/学习/Bioinformatics/QSP/200p_200n/10_fold/' + name + '/km_train/KM_tanimoto_' + name + '_train_' + str(k) + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in K3:
                writer.writerow(row)
            csvfile.close()

        K4 = np.zeros((m, p))
        for i in range(m):
            for j in range(p):
                K4[i][j] = round(metrics_function.tanimoto(X_test[i], X_train[j]), 6)
        print(K4)
        with open('C:/学习/Bioinformatics/QSP/200p_200n/10_fold/' + name + '/km_test/KM_tanimoto_' + name + '_test_' + str(k) + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in K4:
                writer.writerow(row)
            csvfile.close()

