K1 = csvread('D:/Study/Bioinformatics/AFP/kernel_matrix/Antifp_DS2/KM_train_gaussian/KM_gaussian_188-bit_train.csv');
K2 = csvread('D:/Study/Bioinformatics/AFP/kernel_matrix/Antifp_DS2/KM_train_gaussian/KM_gaussian_AAC_train.csv');
K3 = csvread('D:/Study/Bioinformatics/AFP/kernel_matrix/Antifp_DS2/KM_train_gaussian/KM_gaussian_ASDC_train.csv');
K4 = csvread('D:/Study/Bioinformatics/AFP/kernel_matrix/Antifp_DS2/KM_train_gaussian/KM_gaussian_CKSAAP_train.csv');
K5 = csvread('D:/Study/Bioinformatics/AFP/kernel_matrix/Antifp_DS2/KM_train_gaussian/KM_gaussian_DPC_train.csv');
K6 = csvread('D:/Study/Bioinformatics/AFP/kernel_matrix/Antifp_DS2/KM_train_tanimoto/KM_tanimoto_ASDC_train.csv');
K7 = csvread('D:/Study/Bioinformatics/AFP/kernel_matrix/Antifp_DS2/KM_train_tanimoto/KM_tanimoto_CKSAAP_train.csv');
Kernels_list = cat(3, K1, K2, K3, K4, K5, K6, K7);

adjmat = csvread('D:/Study/Bioinformatics/AFP/feature_matrix/Antifp_DS2/train_label.csv');
dim = 1;

regcoef1 = 0.01;
regcoef2 = 0;
[weight_v1] = hsic_kernel_weights_norm(Kernels_list,adjmat,dim,regcoef1,regcoef2)
save('D:/Study/Bioinformatics/AFP/kernel_matrix_G5+T2/Antifp_DS2/KM_train/HSIC_weight.txt', 'weight_v1', '-ascii');