import numpy as np
import csv

f1 = np.loadtxt("C:/学习/Bioinformatics/QSP/121p_179n/kernel_matrix/KM_train_tanimoto/KM_tanimoto_188-bit_train.csv", delimiter = ',')
f2 = np.loadtxt("C:/学习/Bioinformatics/QSP/121p_179n/kernel_matrix/KM_train_tanimoto/KM_tanimoto_AAC_train.csv", delimiter = ',')
f3 = np.loadtxt("C:/学习/Bioinformatics/QSP/121p_179n/kernel_matrix/KM_train_tanimoto/KM_tanimoto_ASDC_train.csv", delimiter = ',')
f4 = np.loadtxt("C:/学习/Bioinformatics/QSP/121p_179n/kernel_matrix/KM_train_tanimoto/KM_tanimoto_CKSAAP_train.csv", delimiter = ',')
f5 = np.loadtxt("C:/学习/Bioinformatics/QSP/121p_179n/kernel_matrix/KM_train_tanimoto/KM_tanimoto_CTD_train.csv", delimiter = ',')

weight_v = np.loadtxt("C:/学习/Bioinformatics/QSP/121p_179n/kernel_matrix/KM_train_tanimoto/weight_tanimoto.txt")

def get_matrix(file):
    m = np.shape(file)[0]
    n = np.shape(file)[1]
    d = np.zeros((m, n))
    for index in range(m):
        d[index] = file[index]
    return d

K1 = get_matrix(f1)
K2 = get_matrix(f2)
K3 = get_matrix(f3)
K4 = get_matrix(f4)
K5 = get_matrix(f5)

K = K1 * weight_v[0] + K2 * weight_v[1] + K3 * weight_v[2] + K4 * weight_v[3] + K5 * weight_v[4]
print(K)

with open('C:/学习/Bioinformatics/QSP/121p_179n/Kernel_matrix/KM_train_tanimoto/combine_tanimoto_train.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in K:
        writer.writerow(row)
    csvfile.close()


t1 = np.loadtxt("C:/学习/Bioinformatics/QSP/121p_179n/Kernel_matrix/KM_test_tanimoto/KM_tanimoto_188-bit_test.csv", delimiter = ',')
t2 = np.loadtxt("C:/学习/Bioinformatics/QSP/121p_179n/Kernel_matrix/KM_test_tanimoto/KM_tanimoto_AAC_test.csv", delimiter = ',')
t3 = np.loadtxt("C:/学习/Bioinformatics/QSP/121p_179n/Kernel_matrix/KM_test_tanimoto/KM_tanimoto_ASDC_test.csv", delimiter = ',')
t4 = np.loadtxt("C:/学习/Bioinformatics/QSP/121p_179n/Kernel_matrix/KM_test_tanimoto/KM_tanimoto_CKSAAP_test.csv", delimiter = ',')
t5 = np.loadtxt("C:/学习/Bioinformatics/QSP/121p_179n/Kernel_matrix/KM_test_tanimoto/KM_tanimoto_CTD_test.csv", delimiter = ',')

weight_v1 = np.loadtxt("C:/学习/Bioinformatics/QSP/121p_179n/Kernel_matrix/KM_test_tanimoto/weight_tanimoto.txt")

G1 = get_matrix(t1)
G2 = get_matrix(t2)
G3 = get_matrix(t3)
G4 = get_matrix(t4)
G5 = get_matrix(t5)

G = G1 * weight_v1[0] + G2 * weight_v1[1] + G3 * weight_v1[2] + G4 * weight_v1[3] + G5 * weight_v1[4]
print(G)

with open('C:/学习/Bioinformatics/QSP/121p_179n/Kernel_matrix/KM_test_tanimoto/combine_tanimoto_test.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in G:
        writer.writerow(row)
    csvfile.close()

