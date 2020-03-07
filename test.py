import numpy as np
import csv


A = np.random.randint(0, 10, (100, 100))
with open('E:/高性能计算实践/test.txt', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in A:
        writer.writerow(row)
csvfile.close()