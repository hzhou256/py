import numpy as np
import csv
import metrics_function

methods_name = ['188-bit', 'AAC', 'ASDC', 'CKSAAP', 'CTD']
for it in range(5):
    name = methods_name[it]
    print(name + ':')
    f1 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/Antifp_Main/' + name + '/test_' + name + '.csv', delimiter = ',', skiprows = 1)
    m = np.shape(f1)[0]
    n = np.shape(f1)[1]
    X_test = np.zeros((m, n-1))
    for index in range(m):
        X_test[index] = f1[index][1:]

    f2 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/Antifp_Main/' + name + '/train_' + name + '.csv', delimiter = ',', skiprows = 1)
    p = np.shape(f2)[0]
    q = np.shape(f2)[1]
    X_train = np.zeros((p, q-1))
    for index in range(p):
        X_train[index] = f2[index][1:]

    K1 = metrics_function.cosine(X_test, X_train)
    print(K1)
    with open('D:/Study/Bioinformatics/AFP/kernel_matrix/Antifp_Main/KM_test_cosine/KM_cosine_' + name + '_test.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in K1:
            writer.writerow(row)
        csvfile.close()

    K3 = metrics_function.tanimoto(X_test, X_train)
    print(K3)
    with open('D:/Study/Bioinformatics/AFP/kernel_matrix/Antifp_Main/KM_test_tanimoto/KM_tanimoto_' + name + '_test.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in K3:
            writer.writerow(row)
        csvfile.close()