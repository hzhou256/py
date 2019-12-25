import numpy as np
import csv
import hsic_kernel_weights_norm


dim = 1
regcoef1 = 0.01
regcoef2 = 0.001

#train kernel cosine
f1 = np.loadtxt('D:/Study/Bioinformatics/AMP/kernel_matrix_4/KM_train_cosine/KM_cosine_AAC_train.csv', delimiter = ',')
f2 = np.loadtxt('D:/Study/Bioinformatics/AMP/kernel_matrix_4/KM_train_cosine/KM_cosine_ASDC_train.csv', delimiter = ',')
f3 = np.loadtxt('D:/Study/Bioinformatics/AMP/kernel_matrix_4/KM_train_cosine/KM_cosine_CKSAAP_train.csv', delimiter = ',')
f4 = np.loadtxt('D:/Study/Bioinformatics/AMP/kernel_matrix_4/KM_train_cosine/KM_cosine_DPC_train.csv', delimiter = ',')
y_train = np.loadtxt('D:/Study/Bioinformatics/AMP/train_label.csv', delimiter = ',')

temp = np.array([f1, f2, f3, f4])
Kernels_list = temp
adjmat = y_train
weight_v = hsic_kernel_weights_norm.hsic_kernel_weights_norm(Kernels_list, adjmat, dim, regcoef1, regcoef2)
print(weight_v)
K = f1 * weight_v[0] + f2 * weight_v[1] + f3 * weight_v[2] + f4 * weight_v[3]
print(K)

with open('D:/Study/Bioinformatics/AMP/kernel_matrix_4/KM_train_cosine/combine_cosine_train.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in K:
        writer.writerow(row)
    csvfile.close()
#test kernel cosine
f1 = np.loadtxt('D:/Study/Bioinformatics/AMP/kernel_matrix_4/KM_test_cosine/KM_cosine_AAC_test.csv', delimiter = ',')
f2 = np.loadtxt('D:/Study/Bioinformatics/AMP/kernel_matrix_4/KM_test_cosine/KM_cosine_ASDC_test.csv', delimiter = ',')
f3 = np.loadtxt('D:/Study/Bioinformatics/AMP/kernel_matrix_4/KM_test_cosine/KM_cosine_CKSAAP_test.csv', delimiter = ',')
f4 = np.loadtxt('D:/Study/Bioinformatics/AMP/kernel_matrix_4/KM_test_cosine/KM_cosine_DPC_test.csv', delimiter = ',')

temp = np.array([f1, f2, f3, f4])
Kernels_list = temp

K = f1 * weight_v[0] + f2 * weight_v[1] + f3 * weight_v[2] + f4 * weight_v[3]
print(K)

with open('D:/Study/Bioinformatics/AMP/kernel_matrix_4/KM_test_cosine/combine_cosine_test.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in K:
        writer.writerow(row)
    csvfile.close()
#train kernel tanimoto
f1 = np.loadtxt('D:/Study/Bioinformatics/AMP/kernel_matrix_4/KM_train_tanimoto/KM_tanimoto_AAC_train.csv', delimiter = ',')
f2 = np.loadtxt('D:/Study/Bioinformatics/AMP/kernel_matrix_4/KM_train_tanimoto/KM_tanimoto_ASDC_train.csv', delimiter = ',')
f3 = np.loadtxt('D:/Study/Bioinformatics/AMP/kernel_matrix_4/KM_train_tanimoto/KM_tanimoto_CKSAAP_train.csv', delimiter = ',')
f4 = np.loadtxt('D:/Study/Bioinformatics/AMP/kernel_matrix_4/KM_train_tanimoto/KM_tanimoto_DPC_train.csv', delimiter = ',')

temp = np.array([f1, f2, f3, f4])
Kernels_list = temp
adjmat = y_train

weight_v = hsic_kernel_weights_norm.hsic_kernel_weights_norm(Kernels_list, adjmat, dim, regcoef1, regcoef2)
print(weight_v)
K = f1 * weight_v[0] + f2 * weight_v[1] + f3 * weight_v[2] + f4 * weight_v[3]
print(K)

with open('D:/Study/Bioinformatics/AMP/kernel_matrix_4/KM_train_tanimoto/combine_tanimoto_train.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in K:
        writer.writerow(row)
    csvfile.close()

#test kernel tanimoto
f1 = np.loadtxt('D:/Study/Bioinformatics/AMP/kernel_matrix_4/KM_test_tanimoto/KM_tanimoto_AAC_test.csv', delimiter = ',')
f2 = np.loadtxt('D:/Study/Bioinformatics/AMP/kernel_matrix_4/KM_test_tanimoto/KM_tanimoto_ASDC_test.csv', delimiter = ',')
f3 = np.loadtxt('D:/Study/Bioinformatics/AMP/kernel_matrix_4/KM_test_tanimoto/KM_tanimoto_CKSAAP_test.csv', delimiter = ',')
f4 = np.loadtxt('D:/Study/Bioinformatics/AMP/kernel_matrix_4/KM_test_tanimoto/KM_tanimoto_DPC_test.csv', delimiter = ',')

temp = np.array([f1, f2, f3, f4])
Kernels_list = temp

K = f1 * weight_v[0] + f2 * weight_v[1] + f3 * weight_v[2] + f4 * weight_v[3]
print(K)

with open('D:/Study/Bioinformatics/AMP/kernel_matrix_4/KM_test_tanimoto/combine_tanimoto_test.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in K:
        writer.writerow(row)
    csvfile.close()

