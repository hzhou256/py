import numpy as np
import csv
import matlab
import matlab.engine

methods_name = ['188-bit', 'AAC', 'ASDC', 'CKSAAP', 'CTD']
l = 2
f1 = np.loadtxt('D:/Study/Bioinformatics/AMP_smote/kernel_matrix_' + str(l+1) + '/KM_train_tanimoto/KM_tanimoto_188-bit_train.csv', delimiter = ',')
f2 = np.loadtxt('D:/Study/Bioinformatics/AMP_smote/kernel_matrix_' + str(l+1) + '/KM_train_tanimoto/KM_tanimoto_AAC_train.csv', delimiter = ',')
f3 = np.loadtxt('D:/Study/Bioinformatics/AMP_smote/kernel_matrix_' + str(l+1) + '/KM_train_tanimoto/KM_tanimoto_ASDC_train.csv', delimiter = ',')
f4 = np.loadtxt('D:/Study/Bioinformatics/AMP_smote/kernel_matrix_' + str(l+1) + '/KM_train_tanimoto/KM_tanimoto_CKSAAP_train.csv', delimiter = ',')
f5 = np.loadtxt('D:/Study/Bioinformatics/AMP_smote/kernel_matrix_' + str(l+1) + '/KM_train_tanimoto/KM_tanimoto_CTD_train.csv', delimiter = ',')

Kernels = np.array([f1, f2, f3, f4, f5])
Kernels_list = matlab.double(Kernels.tolist())

y_label = np.loadtxt('D:/Study/Bioinformatics/AMP_smote/train_label_' + str(l+1) + '.csv', delimiter = ',')
y_label = np.array(y_label)
adjmat = matlab.double(y_label.tolist())
dim = 1
regcoef1 = 0.01
regcoef2 = 0.001

eng = matlab.engine.start_matlab()
weight_v = eng.hsic_kernel_weights_norm(Kernels_list,adjmat,dim,regcoef1,regcoef2)
print(weight_v)

