import numpy as np
import csv
import hsic_kernel_weights_norm


def get_matrix(file):
    m = np.shape(file)[0]
    n = np.shape(file)[1]
    d = np.zeros((m, n))
    for index in range(m):
        d[index] = file[index]
    return d

dataset_name = ['Antifp_Main', 'Antifp_DS1', 'Antifp_DS2']
for ds in range(1):
    name_ds = dataset_name[ds]

    f1 = np.loadtxt('D:/Study/Bioinformatics/AFP/kernel_matrix_auto/' + name_ds +'/KM_train_cosine/KM_cosine_188-bit_train.csv', delimiter = ',')
    f2 = np.loadtxt('D:/Study/Bioinformatics/AFP/kernel_matrix_auto/' + name_ds +'/KM_train_cosine/KM_cosine_AAC_train.csv', delimiter = ',')
    f3 = np.loadtxt('D:/Study/Bioinformatics/AFP/kernel_matrix_auto/' + name_ds +'/KM_train_cosine/KM_cosine_ASDC_train.csv', delimiter = ',')
    f4 = np.loadtxt('D:/Study/Bioinformatics/AFP/kernel_matrix_auto/' + name_ds +'/KM_train_cosine/KM_cosine_CKSAAP_train.csv', delimiter = ',')
    f5 = np.loadtxt('D:/Study/Bioinformatics/AFP/kernel_matrix_auto/' + name_ds +'/KM_train_cosine/KM_cosine_CTD_train.csv', delimiter = ',')
    y_train = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix_auto/Antifp_Main/train_label.csv', delimiter = ',')

    temp = np.array([f1, f2, f3, f4, f5])
    print(np.shape(temp))
    print(np.shape(y_train))
    Kernels_list = temp
    adjmat = y_train
    dim = 1
    regcoef1 = 0.01
    regcoef2 = 0.001

    weight_v = hsic_kernel_weights_norm.hsic_kernel_weights_norm(Kernels_list, adjmat, dim, regcoef1, regcoef2)
    print(weight_v)