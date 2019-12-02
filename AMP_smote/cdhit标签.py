import numpy as np
import csv
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
import pandas as pd


label = []

with open('D:/Study/Bioinformatics/AMP_multiple/label.txt', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        label.append(line)

mb = MultiLabelBinarizer(classes = ['1','2','3','4','5'])
label_mb = mb.fit_transform(label)

f1 = open('D:/Study/Bioinformatics/AMP_multiple/train.fasta')
line = f1.readlines()

seq = []

for line_list in line:
    if not line_list.startswith('>'):
        line_new = line_list.strip('\n')
        seq.append(line_new)


with open('D:/Study/Bioinformatics/AMP_smote/label.csv', 'w', newline='') as csvf2:
    writer = csv.writer(csvf2)
    for row in label_mb:
        writer.writerow(row)
    csvf2.close()

with open('D:/Study/Bioinformatics/AMP_smote/sequence.txt', 'w') as file1:
    for x in seq:
        temp = x + '\n'
        file1.write(temp)
    file1.close()