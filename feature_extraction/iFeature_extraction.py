import os
from scipy.io import arff
import csv
import pandas as pd


os.system('python D:/Study/Bioinformatics/成都培训/git/feature-extraction/iFeature.py --file D:/Study/Bioinformatics/AFP/datasets/Antifp_DS2/train.fasta --out D:/Study/Bioinformatics/AFP/Antifp_DS2/AAC/train_AAC.csv --type AAC')
os.system('python D:/Study/Bioinformatics/成都培训/git/feature-extraction/iFeature.py --file D:/Study/Bioinformatics/AFP/datasets/Antifp_DS2/train.fasta --out D:/Study/Bioinformatics/AFP/Antifp_DS2/CTD/train_CTDC.csv --type CTDC')
os.system('python D:/Study/Bioinformatics/成都培训/git/feature-extraction/iFeature.py --file D:/Study/Bioinformatics/AFP/datasets/Antifp_DS2/train.fasta --out D:/Study/Bioinformatics/AFP/Antifp_DS2/CTD/train_CTDT.csv --type CTDT')
os.system('python D:/Study/Bioinformatics/成都培训/git/feature-extraction/iFeature.py --file D:/Study/Bioinformatics/AFP/datasets/Antifp_DS2/train.fasta --out D:/Study/Bioinformatics/AFP/Antifp_DS2/CTD/train_CTDD.csv --type CTDD')
os.system('python D:/Study/Bioinformatics/成都培训/git/feature-extraction/codes/CKSAAP.py D:/Study/Bioinformatics/AFP/datasets/Antifp_DS2/train.fasta 2 D:/Study/Bioinformatics/AFP/Antifp_DS2/CKSAAP/train_CKSAAP.csv')
os.system('java -jar D:/Study/Bioinformatics/成都培训/188D(SVMProt)/188D.jar D:/Study/Bioinformatics/AFP/datasets/Antifp_DS2/train_negative.fasta D:/Study/Bioinformatics/AFP/Antifp_DS2/188-bit/train_188-bit_negative.arff')
os.system('java -jar D:/Study/Bioinformatics/成都培训/188D(SVMProt)/188D.jar D:/Study/Bioinformatics/AFP/datasets/Antifp_DS2/train_positive.fasta D:/Study/Bioinformatics/AFP/Antifp_DS2/188-bit/train_188-bit_positive.arff')


f1 = pd.read_csv('D:/Study/Bioinformatics/AFP/Antifp_DS2/CTD/train_CTDC.csv', header=None)
f2 = pd.read_csv('D:/Study/Bioinformatics/AFP/Antifp_DS2/CTD/train_CTDT.csv', header=None)
f3 = pd.read_csv('D:/Study/Bioinformatics/AFP/Antifp_DS2/CTD/train_CTDD.csv', header=None)
dat = f1.join(f2.drop(columns=0), lsuffix='_')
dat = dat.join(f3.drop(columns=0), lsuffix='_')
dat.to_csv('D:/Study/Bioinformatics/AFP/Antifp_DS2/CTD/train_CTD.csv', sep=',', header=None, index=False)
