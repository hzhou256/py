import pandas as pd
import csv
import os
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split

'''
ion_frame = pd.read_csv('E:/Study/Bioinformatics/UCI/german/german.csv')
ion_frame.sort_values(by = 'class', inplace = True)
#ion_frame.replace('b', 0, inplace = True)
#ion_frame.replace('g', 1, inplace = True)
class_labels = ion_frame['class'].values
print(class_labels)
#print(ion_frame)
del ion_frame['class']

X_train, X_test, y_train, y_test = train_test_split(ion_frame.values, class_labels, train_size = 700, random_state = 0)
print(y_train, y_test)

X_header = np.zeros((1, np.shape(X_train)[1] + 1))
train = np.column_stack((y_train, X_train))

train = np.row_stack((X_header, train))
test = np.column_stack((y_test, X_test))
test = np.row_stack((X_header, test))

with open('E:/Study/Bioinformatics/UCI/german/X_train.csv', 'w', newline = '') as csvfile:
    writer = csv.writer(csvfile)
    for row in train:
        writer.writerow(row)
    csvfile.close()

with open('E:/Study/Bioinformatics/UCI/german/X_test.csv', 'w', newline = '') as csvfile:
    writer = csv.writer(csvfile)
    for row in test:
        writer.writerow(row)
    csvfile.close()
'''
os.system('python E:/Study/Bioinformatics/FuzzySVM/feature_matrix/CSVtoSVM.py E:/Study/Bioinformatics/UCI/german/X_train.csv E:/Study/Bioinformatics/UCI/german/X_train.svm')
os.system('python E:/Study/Bioinformatics/FuzzySVM/feature_matrix/CSVtoSVM.py E:/Study/Bioinformatics/UCI/german/X_test.csv E:/Study/Bioinformatics/UCI/german/X_test.svm')
