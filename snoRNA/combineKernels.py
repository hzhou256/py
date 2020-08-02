import numpy as np
import csv


names = ['CKSNAP', 'DNC', 'Kmer4', 'Kmer1234', 'NAC', 'RCKmer', 'TNC']

K1 = np.loadtxt('D:/Study/Bioinformatics/snoRNA/tanimoto_kernel/KM_tanimoto_' + names[0] + '_train.csv', delimiter=',')
K2 = np.loadtxt('D:/Study/Bioinformatics/snoRNA/tanimoto_kernel/KM_tanimoto_' + names[1] + '_train.csv', delimiter=',')
K3 = np.loadtxt('D:/Study/Bioinformatics/snoRNA/tanimoto_kernel/KM_tanimoto_' + names[2] + '_train.csv', delimiter=',')
K4 = np.loadtxt('D:/Study/Bioinformatics/snoRNA/tanimoto_kernel/KM_tanimoto_' + names[3] + '_train.csv', delimiter=',')
K5 = np.loadtxt('D:/Study/Bioinformatics/snoRNA/tanimoto_kernel/KM_tanimoto_' + names[4] + '_train.csv', delimiter=',')
K6 = np.loadtxt('D:/Study/Bioinformatics/snoRNA/tanimoto_kernel/KM_tanimoto_' + names[5] + '_train.csv', delimiter=',')
K7 = np.loadtxt('D:/Study/Bioinformatics/snoRNA/tanimoto_kernel/KM_tanimoto_' + names[6] + '_train.csv', delimiter=',')

K = (K1 + K2 + K3 + K4 + K5 + K6 + K7) / 7
with open('D:/Study/Bioinformatics/snoRNA/tanimoto_kernel/KM_tanimoto_AVG_train.csv', 'w',
          newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in K:
        writer.writerow(row)
    csvfile.close()