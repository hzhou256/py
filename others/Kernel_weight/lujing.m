K1 = csvread('D:/Study/Bioinformatics/补实验/AFP/kernel_matrix/Antifp_DS2/K_train_188-bit.csv');
K2 = csvread('D:/Study/Bioinformatics/补实验/AFP/kernel_matrix/Antifp_DS2/K_train_AAC.csv');
K3 = csvread('D:/Study/Bioinformatics/补实验/AFP/kernel_matrix/Antifp_DS2/K_train_ASDC.csv');
K4 = csvread('D:/Study/Bioinformatics/补实验/AFP/kernel_matrix/Antifp_DS2/K_train_CKSAAP.csv');
K5 = csvread('D:/Study/Bioinformatics/补实验/AFP/kernel_matrix/Antifp_DS2/K_train_DPC.csv');

Kernels_list = cat(3, K1, K2, K3, K4, K5);

adjmat = csvread('D:/Study/Bioinformatics/补实验/AFP/feature_matrix/Antifp_DS2/train_label.csv');
dim = 1;

%regcoef1 = 0.01;
%regcoef2 = 0;
%[weight_v1] = hsic_kernel_weights_norm(Kernels_list,adjmat,dim,regcoef1,regcoef2)
%save('D:/Study/Bioinformatics/AFP/kernel_matrix_CKSAAP+DPC/Antifp_DS2/KM_train_tanimoto/HSIC_weight.txt', 'weight_v1', '-ascii');

r_lamda = 0.1 * trace(Kernels_list(:,:,1) * Kernels_list(:,:,1)');
[weight_v2] = FKL_weights(Kernels_list,adjmat,dim,r_lamda)
%save('D:/Study/Bioinformatics/AFP/kernel_matrix_CKSAAP+DPC/Antifp_DS2/KM_train_tanimoto/FKL_weight.txt', 'weight_v2', '-ascii');

[weight_v3] = TKA_kernels_weights(Kernels_list,adjmat,dim)
%save('D:/Study/Bioinformatics/AFP/kernel_matrix_CKSAAP+DPC/Antifp_DS2/KM_train_tanimoto/TKA_weight.txt', 'weight_v3', '-ascii');