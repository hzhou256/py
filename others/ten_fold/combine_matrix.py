import numpy as np
import csv

methods_name = ['188-bit', 'AAC', 'ASDC', 'CKSAAP', 'CTD']
kernel_name = ['cosine', 'tanimoto']

def get_matrix(file):
    m = np.shape(file)[0]
    n = np.shape(file)[1]
    d = np.zeros((m, n))
    for index in range(m):
        d[index] = file[index]
    return d

for it in range(2):
    s = kernel_name[it]
    for i in range(10):
        f1 = np.loadtxt('D:/study/Bioinformatics/QSP/200p_200n/10_fold/188-bit/km_train/KM_' + s + '_188-bit_train_' + str(i) + '.csv', delimiter = ',')
        f2 = np.loadtxt('D:/study/Bioinformatics/QSP/200p_200n/10_fold/AAC/km_train/KM_' + s + '_AAC_train_' + str(i) + '.csv', delimiter = ',')
        f3 = np.loadtxt('D:/study/Bioinformatics/QSP/200p_200n/10_fold/ASDC/km_train/KM_' + s + '_ASDC_train_' + str(i) + '.csv', delimiter = ',')
        f4 = np.loadtxt('D:/study/Bioinformatics/QSP/200p_200n/10_fold/CKSAAP/km_train/KM_' + s + '_CKSAAP_train_' + str(i) + '.csv', delimiter = ',')
        f5 = np.loadtxt('D:/study/Bioinformatics/QSP/200p_200n/10_fold/CTD/km_train/KM_' + s + '_CTD_train_' + str(i) + '.csv', delimiter = ',')
        weight_v = np.loadtxt('D:/study/Bioinformatics/QSP/200p_200n/kernel_matrix/KM_train_' + s + '/weight_' + s + '.txt')
        K1 = get_matrix(f1)
        K2 = get_matrix(f2)
        K3 = get_matrix(f3)
        K4 = get_matrix(f4)
        K5 = get_matrix(f5)
        K = K1 * weight_v[0] + K2 * weight_v[1] + K3 * weight_v[2] + K4 * weight_v[3] + K5 * weight_v[4]
        print(K)
        with open('D:/study/Bioinformatics/QSP/200p_200n/10_fold/combine_matrix/combine_' + s + '_train_' + str(i) + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in K:
                writer.writerow(row)
            csvfile.close()

        t1 = np.loadtxt('D:/study/Bioinformatics/QSP/200p_200n/10_fold/188-bit/km_test/KM_' + s + '_188-bit_test_' + str(i) + '.csv', delimiter = ',')
        t2 = np.loadtxt('D:/study/Bioinformatics/QSP/200p_200n/10_fold/AAC/km_test/KM_' + s + '_AAC_test_' + str(i) + '.csv', delimiter = ',')
        t3 = np.loadtxt('D:/study/Bioinformatics/QSP/200p_200n/10_fold/ASDC/km_test/KM_' + s + '_ASDC_test_' + str(i) + '.csv', delimiter = ',')
        t4 = np.loadtxt('D:/study/Bioinformatics/QSP/200p_200n/10_fold/CKSAAP/km_test/KM_' + s + '_CKSAAP_test_' + str(i) + '.csv', delimiter = ',')
        t5 = np.loadtxt('D:/study/Bioinformatics/QSP/200p_200n/10_fold/CTD/km_test/KM_' + s + '_CTD_test_' + str(i) + '.csv', delimiter = ',')
        G1 = get_matrix(t1)
        G2 = get_matrix(t2)
        G3 = get_matrix(t3)
        G4 = get_matrix(t4)
        G5 = get_matrix(t5)
        G = G1 * weight_v[0] + G2 * weight_v[1] + G3 * weight_v[2] + G4 * weight_v[3] + G5 * weight_v[4]
        print(G)
        with open('D:/study/Bioinformatics/QSP/200p_200n/10_fold/combine_matrix/combine_' + s + '_test_' + str(i) + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in G:
                writer.writerow(row)
            csvfile.close()