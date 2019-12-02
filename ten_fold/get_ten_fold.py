import numpy as np
import csv

methods_name = ['188-bit', 'AAC', 'ASDC', 'CKSAAP', 'CTD']
for it in range(5):
    name = methods_name[it]
    print(name + ':')

    f1 = np.loadtxt('C:/学习/Bioinformatics/QSP/200p_200n/10_fold/' + name + '/train_' + name + '.csv', delimiter = ',', skiprows = 1)
    m = 0
    n = 20
    k = 0
    for k in range(10):
        index = 0
        temp = np.zeros((40, np.shape(f1)[1]))
        for i in range(m, n):
            temp[index] = f1[i]
            index = index + 1
        for j in range(m, n):
            temp[index] = f1[j+200]
            index = index + 1
        m = m + 20
        n = n + 20
        with open('C:/学习/Bioinformatics/QSP/200p_200n/10_fold/' + name + '/test/test_' + name + '_' + str(k) + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in temp:
                writer.writerow(row)
            csvfile.close()
    m = 0
    n = 20
    k = 0
    for k in range(10):
        temp = np.copy(f1)
        temp = np.delete(temp, list(range(m, n)) + list(range(m+200, n+200)), axis = 0)
        m = m + 20
        n = n + 20
        with open('C:/学习/Bioinformatics/QSP/200p_200n/10_fold/' + name + '/train/train_' + name + '_' + str(k) + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in temp:
                writer.writerow(row)
            csvfile.close()    


