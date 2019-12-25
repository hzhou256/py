K1 = csvread('D:/Study/Bioinformatics/AFP/kernel_matrix/Antifp_DS1/KM_train_tanimoto/KM_tanimoto_188-bit_train.csv');
K2 = csvread('D:/Study/Bioinformatics/AFP/kernel_matrix/Antifp_DS1/KM_train_tanimoto/KM_tanimoto_AAC_train.csv');
K3 = csvread('D:/Study/Bioinformatics/AFP/kernel_matrix/Antifp_DS1/KM_train_tanimoto/KM_tanimoto_ASDC_train.csv');
K4 = csvread('D:/Study/Bioinformatics/AFP/kernel_matrix/Antifp_DS1/KM_train_tanimoto/KM_tanimoto_CKSAAP_train.csv');
K5 = csvread('D:/Study/Bioinformatics/AFP/kernel_matrix/Antifp_DS1/KM_train_tanimoto/KM_tanimoto_CTD_train.csv');
K6 = csvread('D:/Study/Bioinformatics/AFP/kernel_matrix/Antifp_DS1/KM_train_tanimoto/KM_tanimoto_DPC_train.csv');
%Kernels_list = cat(3, K1, K2, K3, K4, K5, K6);
Kernels_list = cat(3, K1, K2, K3, K4, K6);
adjmat = csvread('D:/Study/Bioinformatics/AFP/feature_matrix/Antifp_DS1/train_label.csv');
dim = 1;
r_lamda = 20000;
[weight_v] = FKL_weights(Kernels_list,adjmat,dim,r_lamda)
save('D:/Study/Bioinformatics/AFP/kernel_matrix_5/Antifp_DS1/KM_train_tanimoto/FKL_weight.txt', 'weight_v', '-ascii');