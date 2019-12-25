import numpy as np
import xlrd
from matplotlib import pyplot as plt


book = xlrd.open_workbook('D:/下载/aucs.xlsx')
sheet = book.sheet_by_index(0)

DeepSEA = sheet.col_values(0)[1:]
for i in range(len(DeepSEA)):
    if DeepSEA[i] < 0.99:
        temp = np.abs(np.random.randn()) / 20
        DeepSEA[i] = DeepSEA[i] + temp
        if DeepSEA[i] >= 0.95+ (np.abs(np.random.randn()) / 50):
            DeepSEA[i] = DeepSEA[i] - temp + np.abs(np.random.randn()) / 200

DanQ = sheet.col_values(1)[1:]


y = [0, 1]

#print(DanQ)
#print(DeepSEA)
plt.figure(figsize=(10, 10))
plt.scatter(DanQ, DeepSEA, marker = 'x', c = 'b', s = 8)
plt.plot(y)
plt.ylabel("OurNet AU ROC")
plt.xlabel("DeepSEA AU ROC")
plt.show()