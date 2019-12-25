import numpy as np
import csv


dataset_name = ['Antifp_Main', 'Antifp_DS1', 'Antifp_DS2']
mkl_name = ['HSIC', 'FKL', 'TKA', 'HKA', 'CKA']
for ds in range(3):
    name_ds = dataset_name[ds]
    print(name_ds)
    for mkl in range(1):
        name_mkl = mkl_name[mkl]
        print('MKL: ', name_mkl)
        #train kernel tanimoto
        f1 = np.loadtxt('D:/Study/Bioinformatics/AFP/kernel_matrix/' + name_ds +'/KM_train_tanimoto/KM_tanimoto_ASDC_train.csv', delimiter = ',')
        f2 = np.loadtxt('D:/Study/Bioinformatics/AFP/kernel_matrix/' + name_ds +'/KM_train_tanimoto/KM_tanimoto_CKSAAP_train.csv', delimiter = ',')

        weight_v = np.loadtxt('D:/Study/Bioinformatics/AFP/kernel_matrix_CKSAAP+DPC/' + name_ds +'/KM_train_tanimoto/' + name_mkl + '_weight.txt', delimiter = ',', usecols = 0)
        print(weight_v)
        K = f1 * weight_v[0] + f2 * weight_v[1]
        print(K)

        with open('D:/Study/Bioinformatics/AFP/kernel_matrix_CKSAAP+DPC/' + name_ds +'/KM_train_tanimoto/combine_tanimoto_' + name_mkl + '_train.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in K:
                writer.writerow(row)
            csvfile.close()
        #test kernel tanimoto
        f1 = np.loadtxt('D:/Study/Bioinformatics/AFP/kernel_matrix/' + name_ds +'/KM_test_tanimoto/KM_tanimoto_ASDC_test.csv', delimiter = ',')
        f2 = np.loadtxt('D:/Study/Bioinformatics/AFP/kernel_matrix/' + name_ds +'/KM_test_tanimoto/KM_tanimoto_CKSAAP_test.csv', delimiter = ',')

        K = f1 * weight_v[0] + f2 * weight_v[1]
        print(K)

        with open('D:/Study/Bioinformatics/AFP/kernel_matrix_CKSAAP+DPC/' + name_ds +'/KM_test_tanimoto/combine_tanimoto_' + name_mkl + '_test.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in K:
                writer.writerow(row)
            csvfile.close()

