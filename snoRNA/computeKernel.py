import csv
import numpy as np
from scipy.spatial.distance import cdist


def tanimoto_base(p_vec, q_vec):
    pq = np.dot(p_vec, q_vec)
    p_square = np.square(np.linalg.norm(p_vec))
    q_square = np.square(np.linalg.norm(q_vec))
    d = pq / (p_square + q_square - pq)
    return d


def tanimoto(X, Y):
    K = cdist(X, Y, tanimoto_base)
    return K


dataset = ['CKSNAP', 'DNC', 'Kmer4', 'Kmer1234', 'NAC', 'RCKmer', 'TNC']
for i in range(0, 7):
    name = dataset[i]
    print(name)

    f1 = np.loadtxt('D:/Study/Bioinformatics/snoRNA/feature/snoRNA_' + name + '.csv',
                    delimiter=',')
    X = f1
    K = tanimoto(X, X)
    with open('D:/Study/Bioinformatics/snoRNA/tanimoto_kernel/KM_tanimoto_' + name + '_train.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in K:
            writer.writerow(row)
        csvfile.close()
