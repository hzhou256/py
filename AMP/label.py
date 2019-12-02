import numpy as np
import csv
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier

label = []

with open('C:/学习/Bioinformatics/AMP_multiple/label.txt', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        label.append(line)

mb = MultiLabelBinarizer(classes = ['1','2','3','4','5'])
label_mb = mb.fit_transform(label)
print(label_mb)
