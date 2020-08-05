import csv
from sklearn.metrics import pairwise
import numpy as np

file_name = "mRNA"
dataset = ['CKSNAP', 'Kmer4', 'Kmer1234', 'NAC', 'RCKmer', 'DNC', 'TNC']

gamma_list_mRNA = [2.979166667, 1, 3.5625, 16.89930556, 2.045277778, 12.89583333, 2.670138889]

if file_name == "mRNA":
    gamma_list = gamma_list_mRNA

for i in range(0, 7):
    name = dataset[i]
    print(name)

    f1 = np.loadtxt(
        'D:/Study/Bioinformatics/王浩/data and code/data/feature/' + file_name + '/' + file_name + '_' + name + '.csv',
        delimiter=',')

    X = f1

    K = pairwise.rbf_kernel(X, gamma=gamma_list[i])
    with open('D:/Study/Bioinformatics/mRNA/rbf_kernel/KM_rbf_' + name + '_train.csv', 'w',
              newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in K:
            writer.writerow(row)
        csvfile.close()
