import numpy as np
import csv
import metrics_function

methods_name = ['188-bit', 'AAC', 'ASDC', 'CKSAAP', 'CTD']
gamma_list = [1.5161741149136382e-07, 0.5, 0.5, 0.5, 0.5]
for it in range(5):
    name = methods_name[it]
    print(name + ':')
    file = np.loadtxt('D:/Study/Bioinformatics/AMP/' + name + '/train_' + name + '.csv', delimiter = ',', skiprows = 1)
    m = np.shape(file)[0]
    n = np.shape(file)[1]
    data = np.zeros((m, n-1))
    for index in range(m):
        data[index] = file[index][1:]

    K = np.zeros((m, m))
    gamma = gamma_list[it]
    np.set_printoptions(suppress = True)
    for i in range(m):
        for j in range(m):
            K[i][j] = round(metrics_function.gaussian(data[i], data[j], gamma), 6)
    print(K)
    with open('D:/Study/Bioinformatics/AMP/kernel_matrix/KM_train_gaussian/KM_gaussian_' + name + '_train.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in K:
            writer.writerow(row)
        csvfile.close()

    K1 = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            K1[i][j] = round(metrics_function.cosine(data[i], data[j]), 6)
    print(K1)
    with open('D:/Study/Bioinformatics/AMP/kernel_matrix/KM_train_cosine/KM_cosine_' + name + '_train.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in K1:
            writer.writerow(row)
        csvfile.close()

    K2 = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            K2[i][j] = round(metrics_function.pearson(data[i], data[j]), 6)
    print(K2)
    with open('D:/Study/Bioinformatics/AMP/kernel_matrix/KM_train_pearson/KM_pearson_' + name + '_train.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in K2:
            writer.writerow(row)
        csvfile.close()

    K3 = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            K3[i][j] = round(metrics_function.tanimoto(data[i], data[j]), 6)
    print(K3)
    with open('D:/Study/Bioinformatics/AMP/kernel_matrix/KM_train_tanimoto/KM_tanimoto_' + name + '_train.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in K3:
            writer.writerow(row)
        csvfile.close()