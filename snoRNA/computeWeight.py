import numpy as np
from snoRNA.hsic_kernel_weights_norm import hsic_kernel_weights_norm


names = ['CKSNAP', 'DNC', 'Kmer4', 'Kmer1234', 'NAC', 'RCKmer', 'TNC']

K1 = np.loadtxt('D:/Study/Bioinformatics/snoRNA/tanimoto_kernel/KM_tanimoto_' + names[0] + '_train.csv', delimiter=',')
K2 = np.loadtxt('D:/Study/Bioinformatics/snoRNA/tanimoto_kernel/KM_tanimoto_' + names[1] + '_train.csv', delimiter=',')
K3 = np.loadtxt('D:/Study/Bioinformatics/snoRNA/tanimoto_kernel/KM_tanimoto_' + names[2] + '_train.csv', delimiter=',')
K4 = np.loadtxt('D:/Study/Bioinformatics/snoRNA/tanimoto_kernel/KM_tanimoto_' + names[3] + '_train.csv', delimiter=',')
K5 = np.loadtxt('D:/Study/Bioinformatics/snoRNA/tanimoto_kernel/KM_tanimoto_' + names[4] + '_train.csv', delimiter=',')
K6 = np.loadtxt('D:/Study/Bioinformatics/snoRNA/tanimoto_kernel/KM_tanimoto_' + names[5] + '_train.csv', delimiter=',')
K7 = np.loadtxt('D:/Study/Bioinformatics/snoRNA/tanimoto_kernel/KM_tanimoto_' + names[6] + '_train.csv', delimiter=',')
label = np.loadtxt('D:/Study/Bioinformatics/snoRNA/snoRNA_label.csv', delimiter=',', skiprows=1)

K_train = np.array([K1, K2, K3, K4, K5, K6, K7])

kernel_weights = hsic_kernel_weights_norm(K_train, label, 1, 0.1, 0.01)
print(kernel_weights)
