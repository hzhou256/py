import numpy as np
import csv
import metrics_function

methods_name = ['188-bit', 'AAC', 'ASDC', 'CKSAAP', 'CTD']
for it in range(5):
    name = methods_name[it]
    print(name + ':')
    file = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/Antifp_Main/' + name + '/train_' + name + '.csv', delimiter = ',', skiprows = 1)
    m = np.shape(file)[0]
    n = np.shape(file)[1]
    data = np.zeros((m, n-1))
    for index in range(m):
        data[index] = file[index][1:]

    np.set_printoptions(suppress = True)

    K1 = metrics_function.cosine(data, data)
    print(K1)
    with open('D:/Study/Bioinformatics/AFP/kernel_matrix/Antifp_Main/KM_train_cosine/KM_cosine_' + name + '_train.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in K1:
            writer.writerow(row)
        csvfile.close()

    K3 = metrics_function.tanimoto(data, data)
    print(K3)
    with open('D:/Study/Bioinformatics/AFP/kernel_matrix/Antifp_Main/KM_train_tanimoto/KM_tanimoto_' + name + '_train.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in K3:
            writer.writerow(row)
        csvfile.close()