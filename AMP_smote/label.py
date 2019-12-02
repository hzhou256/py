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

print(label)
sequence = dict()
with open('D:/Study/Bioinformatics/AMP_multiple/train.fasta', 'r') as f:
    line = f.readlines()
    l = len(line)
    i = 0
    k = 0
    while i < l:
        if line[i].startswith('>'):
            s = line[i+1].strip('/n')
            temp = {s: label[k]}
            k = k + 1
            sequence.update(temp)
        i = i + 1

fseq = open("D:/Study/Bioinformatics/AMP_smote/class_dict_.fasta","w")
for line in list(sequence):
    fseq.write(line)
fseq.close()

for c in range(5):
    temp = []
    for key, value in sequence.items():
        if str(c+1) in value:
            temp.append(key)
    f1 = open("D:/Study/Bioinformatics/AMP_smote/class_" + str(c+1) + "_.fasta","w")
    k = 1
    for line in temp:
        f1.write('>' + str(k) + '|' + str(c+1) + '\n')
        k = k + 1
        f1.write(line)
    f1.close()