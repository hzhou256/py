import os
import csv
import pandas as pd


os.system('python E:/Study/iFeature/iFeature.py --file E:/Study/Bioinformatics/AMP/fasta/test.fasta --out E:/Study/Bioinformatics/AMP/features/AAC/test_AAC.csv --type AAC')
#os.system('python D:/Study/iFeature/iFeature.py --file D:/Study/Bioinformatics/AMP/test.fasta --out D:/Study/Bioinformatics/AMP/CTD/test_CTDC.csv --type CTDC')
#os.system('python D:/Study/iFeature/iFeature.py --file D:/Study/Bioinformatics/AMP/test.fasta --out D:/Study/Bioinformatics/AMP/CTD/test_CTDT.csv --type CTDT')
#os.system('python D:/Study/iFeature/iFeature.py --file D:/Study/Bioinformatics/AMP/test.fasta --out D:/Study/Bioinformatics/AMP/CTD/test_CTDD.csv --type CTDD')
os.system('python E:/Study/iFeature/codes/CKSAAP.py E:/Study/Bioinformatics/AMP/fasta/test.fasta 2 E:/Study/Bioinformatics/AMP/features/CKSAAP/test_CKSAAP.csv')
#os.system('java -jar D:/Study/Bioinformatics/成都培训/188D(SVMProt)/188D.jar D:/Study/Bioinformatics/AMP/test_negative.fasta D:/Study/Bioinformatics/AMP/188-bit/test_188-bit_negative.arff')
#os.system('java -jar D:/Study/Bioinformatics/成都培训/188D(SVMProt)/188D.jar D:/Study/Bioinformatics/AMP/test_positive.fasta D:/Study/Bioinformatics/AMP/188-bit/test_188-bit_positive.arff')


#f1 = pd.read_csv('D:/Study/Bioinformatics/AMP/CTD/test_CTDC.csv', header=None)
#f2 = pd.read_csv('D:/Study/Bioinformatics/AMP/CTD/test_CTDT.csv', header=None)
#f3 = pd.read_csv('D:/Study/Bioinformatics/AMP/CTD/test_CTDD.csv', header=None)
#dat = f1.join(f2.drop(columns=0), lsuffix='_')
#dat = dat.join(f3.drop(columns=0), lsuffix='_')
#dat.to_csv('D:/Study/Bioinformatics/AMP/CTD/test_CTD.csv', sep=',', header=None, index=False)
