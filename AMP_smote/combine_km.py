import numpy as np
import csv

f1 = np.loadtxt("D:/Study/Bioinformatics/AMP_smote/kernel_matrix_3/KM_train_cosine/KM_cosine_188-bit_train.csv", delimiter = ',')
f2 = np.loadtxt("D:/Study/Bioinformatics/AMP_smote/kernel_matrix_3/KM_train_cosine/KM_cosine_AAC_train.csv", delimiter = ',')
f3 = np.loadtxt("D:/Study/Bioinformatics/AMP_smote/kernel_matrix_3/KM_train_cosine/KM_cosine_ASDC_train.csv", delimiter = ',')
f4 = np.loadtxt("D:/Study/Bioinformatics/AMP_smote/kernel_matrix_3/KM_train_cosine/KM_cosine_CKSAAP_train.csv", delimiter = ',')
f5 = np.loadtxt("D:/Study/Bioinformatics/AMP_smote/kernel_matrix_3/KM_train_cosine/KM_cosine_CTD_train.csv", delimiter = ',')

weight_v = np.loadtxt("D:/Study/Bioinformatics/AMP_smote/kernel_matrix_3/KM_train_cosine/weight_cosine.txt")

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

with open('D:/Study/Bioinformatics/AMP_smote/kernel_matrix_3/KM_train_cosine/combine_cosine_train.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in K:
        writer.writerow(row)
    csvfile.close()
