import numpy as np
import csv
import metrics_function

methods_name = ['188-bit', 'AAC', 'ASDC', 'CKSAAP', 'CTD']
gamma_list = [1.0563062941328463e-05, 0.5, 0.5, 0.5, 0.5]
for it in range(5):
    name = methods_name[it]
    print(name + ':')
    f1 = np.loadtxt('D:/Study/Bioinformatics/AMP/' + name + '/test_' + name + '.csv', delimiter = ',', skiprows = 1)
    m = np.shape(f1)[0]
    n = np.shape(f1)[1]
    X_test = np.zeros((m, n-1))
    for index in range(m):
        X_test[index] = f1[index][1:]

    f2 = np.loadtxt('D:/Study/Bioinformatics/AMP/' + name + '/train_' + name + '.csv', delimiter = ',', skiprows = 1)
    p = np.shape(f2)[0]
    q = np.shape(f2)[1]
    X_train = np.zeros((p, q-1))
    for index in range(p):
        X_train[index] = f2[index][1:]

    K = np.zeros((m, p))
    gamma = gamma_list[it]
    np.set_printoptions(suppress = True)
    for i in range(m):
        for j in range(p):
            K[i][j] = round(metrics_function.gaussian(X_test[i], X_train[j], gamma), 6)
    print(K)
    with open('D:/Study/Bioinformatics/AMP/kernel_matrix/KM_test_gaussian/KM_gaussian_' + name + '_test.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in K:
            writer.writerow(row)
        csvfile.close()

    K1 = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            K1[i][j] = round(metrics_function.cosine(X_test[i], X_train[j]), 6)
    print(K1)
    with open('D:/Study/Bioinformatics/AMP/kernel_matrix/KM_test_cosine/KM_cosine_' + name + '_test.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in K1:
            writer.writerow(row)
        csvfile.close()

    K2 = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            K2[i][j] = round(metrics_function.pearson(X_test[i], X_train[j]), 6)
    print(K2)
    with open('D:/Study/Bioinformatics/AMP/kernel_matrix/KM_test_pearson/KM_pearson_' + name + '_test.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in K2:
            writer.writerow(row)
        csvfile.close()

    K3 = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            K3[i][j] = round(metrics_function.tanimoto(X_test[i], X_train[j]), 6)
    print(K3)
    with open('D:/Study/Bioinformatics/AMP/kernel_matrix/KM_test_tanimoto/KM_tanimoto_' + name + '_test.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in K3:
            writer.writerow(row)
        csvfile.close()