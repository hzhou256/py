import numpy as np
import csv


dataset_name = ['Antifp_Main', 'Antifp_DS1', 'Antifp_DS2']
mkl_name = ['HSIC', 'FKL', 'TKA', 'HKA', 'CKA']
for ds in range(3):
    name_ds = dataset_name[ds]
    print(name_ds)
    for mkl in range(0,3):
        name_mkl = mkl_name[mkl]
        print('MKL: ', name_mkl)
        #train kernel
        f1 = np.loadtxt('D:/Study/Bioinformatics/AFP/kernel_matrix/' + name_ds +'/KM_train_gaussian/KM_gaussian_188-bit_train.csv', delimiter = ',')       
        f2 = np.loadtxt('D:/Study/Bioinformatics/AFP/kernel_matrix/' + name_ds +'/KM_train_gaussian/KM_gaussian_AAC_train.csv', delimiter = ',')
        f3 = np.loadtxt('D:/Study/Bioinformatics/AFP/kernel_matrix/' + name_ds +'/KM_train_gaussian/KM_gaussian_ASDC_train.csv', delimiter = ',')
        f4 = np.loadtxt('D:/Study/Bioinformatics/AFP/kernel_matrix/' + name_ds +'/KM_train_gaussian/KM_gaussian_CKSAAP_train.csv', delimiter = ',')
        f5 = np.loadtxt('D:/Study/Bioinformatics/AFP/kernel_matrix/' + name_ds +'/KM_train_gaussian/KM_gaussian_DPC_train.csv', delimiter = ',')      

        weight_v = np.loadtxt('D:/Study/Bioinformatics/AFP/kernel_matrix_5/' + name_ds +'/KM_train_gaussian/' + name_mkl + '_weight.txt', delimiter = ',', usecols = 0)
        print(weight_v)
        K = f1 * weight_v[0] + f2 * weight_v[1] + f3 * weight_v[2] + f4 * weight_v[3] + f5 * weight_v[4]
        print(K)

        with open('D:/Study/Bioinformatics/AFP/kernel_matrix_5/' + name_ds +'/KM_train_gaussian/combine_gaussian_' + name_mkl + '_train.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in K:
                writer.writerow(row)
            csvfile.close()
        #test kernel
        f1 = np.loadtxt('D:/Study/Bioinformatics/AFP/kernel_matrix/' + name_ds +'/KM_test_gaussian/KM_gaussian_188-bit_test.csv', delimiter = ',')        
        f2 = np.loadtxt('D:/Study/Bioinformatics/AFP/kernel_matrix/' + name_ds +'/KM_test_gaussian/KM_gaussian_AAC_test.csv', delimiter = ',')
        f3 = np.loadtxt('D:/Study/Bioinformatics/AFP/kernel_matrix/' + name_ds +'/KM_test_gaussian/KM_gaussian_ASDC_test.csv', delimiter = ',')
        f4 = np.loadtxt('D:/Study/Bioinformatics/AFP/kernel_matrix/' + name_ds +'/KM_test_gaussian/KM_gaussian_CKSAAP_test.csv', delimiter = ',')
        f5 = np.loadtxt('D:/Study/Bioinformatics/AFP/kernel_matrix/' + name_ds +'/KM_test_gaussian/KM_gaussian_DPC_test.csv', delimiter = ',')
        K = f1 * weight_v[0] + f2 * weight_v[1] + f3 * weight_v[2] + f4 * weight_v[3] + f5 * weight_v[4]
        print(K)

        with open('D:/Study/Bioinformatics/AFP/kernel_matrix_5/' + name_ds +'/KM_test_gaussian/combine_gaussian_' + name_mkl + '_test.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in K:
                writer.writerow(row)
            csvfile.close()

