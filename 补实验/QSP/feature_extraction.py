import os
import csv
import pandas as pd


#os.system('python D:/Study/Bioinformatics/成都培训/git/feature-extraction/iFeature.py --file D:/Study/Bioinformatics/补实验/QSP/fasta/train.fasta --out D:/Study/Bioinformatics/补实验/QSP/features/train_AAC.csv --type AAC')
#os.system('python D:/Study/Bioinformatics/成都培训/git/feature-extraction/iFeature.py --file D:/Study/Bioinformatics/补实验/QSP/fasta/710/710.fasta --out D:/Study/Bioinformatics/补实验/QSP/features/710/710_AAC.csv --type AAC')

#os.system('python D:/Study/Bioinformatics/成都培训/git/feature-extraction/iFeature.py --file D:/Study/Bioinformatics/补实验/QSP/fasta/train.fasta --out D:/Study/Bioinformatics/补实验/QSP/features/train_DPC.csv --type DPC')
#os.system('python D:/Study/Bioinformatics/成都培训/git/feature-extraction/iFeature.py --file D:/Study/Bioinformatics/补实验/QSP/fasta/710/710.fasta --out D:/Study/Bioinformatics/补实验/QSP/features/710/710_DPC.csv --type DPC')

#os.system('python D:/Study/Bioinformatics/成都培训/git/feature-extraction/codes/CKSAAP.py D:/Study/Bioinformatics/补实验/QSP/fasta/train.fasta 3 D:/Study/Bioinformatics/补实验/QSP/features/train_CKSAAP.csv')
#os.system('python D:/Study/Bioinformatics/成都培训/git/feature-extraction/codes/CKSAAP.py D:/Study/Bioinformatics/补实验/QSP/fasta/710/710.fasta 3 D:/Study/Bioinformatics/补实验/QSP/features/710/710_CKSAAP.csv')

os.system('java -jar D:/Study/Bioinformatics/成都培训/188D(SVMProt)/188D.jar D:\\Study\\Bioinformatics\\补实验\\QSP\\fasta\\train_pos.fasta D:\\Study\\Bioinformatics\\补实验\\QSP\\features\\train_188-bit_pos.arff')
os.system('java -jar D:/Study/Bioinformatics/成都培训/188D(SVMProt)/188D.jar D:\\Study\\Bioinformatics\\补实验\\QSP\\fasta\\train_neg.fasta D:\\Study\\Bioinformatics\\补实验\\QSP\\features\\train_188-bit_neg.arff')
#os.system('java -jar D:/Study/Bioinformatics/成都培训/188D(SVMProt)/188D.jar D:\\Study\\Bioinformatics\\补实验\\QSP\\fasta\\test_pos.fasta D:\\Study\\Bioinformatics\\补实验\\QSP\\features\\188\\test_188-bit_pos.arff')
#os.system('java -jar D:/Study/Bioinformatics/成都培训/188D(SVMProt)/188D.jar D:\\Study\\Bioinformatics\\补实验\\QSP\\fasta\\test_neg.fasta D:\\Study\\Bioinformatics\\补实验\\QSP\\features\\188\\test_188-bit_neg.arff')

