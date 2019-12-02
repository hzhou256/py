import numpy as np
import csv
import metrics_function

methods_name = ['188-bit', 'AAC', 'ASDC', 'CKSAAP', 'CTD']
for it in range(5):
    name = methods_name[it]
    print(name + ':')
    f1 = np.loadtxt('C:/学习/Bioinformatics/AMP/' + name + '/test_' + name + '.csv', delimiter = ',', skiprows = 1)
    m = np.shape(f1)[0]
    n = np.shape(f1)[1]
    X_test = np.zeros((m, n-1))
    for index in range(m):
        X_test[index] = f1[index][1:]

    f2 = np.loadtxt('C:/学习/Bioinformatics/AMP/' + name + '/train_' + name + '.csv', delimiter = ',', skiprows = 1)
    p = np.shape(f2)[0]
    q = np.shape(f2)[1]
    X_train = np.zeros((p, q-1))
    for index in range(p):
        X_train[index] = f2[index][1:]

    K1 = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            K1[i][j] = round(metrics_function.cosine(X_test[i], X_train[j]), 6)
    print(K1)
    with open('C:/学习/Bioinformatics/AMP/kernel_matrix/KM_test_cosine/KM_cosine_' + name + '_test.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in K1:
            writer.writerow(row)
        csvfile.close()

    K3 = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            K3[i][j] = round(metrics_function.tanimoto(X_test[i], X_train[j]), 6)
    print(K3)
    with open('C:/学习/Bioinformatics/AMP/kernel_matrix/KM_test_tanimoto/KM_tanimoto_' + name + '_test.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in K3:
            writer.writerow(row)
        csvfile.close()