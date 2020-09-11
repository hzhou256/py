import os
import csv
import pandas as pd


os.system('python D:/Study/Bioinformatics/成都培训/git/feature-extraction/iFeature.py --file D:/Study/Bioinformatics/补实验/AIE/fasta/second/train.fasta --out D:/Study/Bioinformatics/补实验/AIE/features/second/train_AAC.csv --type AAC')
os.system('python D:/Study/Bioinformatics/成都培训/git/feature-extraction/iFeature.py --file D:/Study/Bioinformatics/补实验/AIE/fasta/second/test.fasta --out D:/Study/Bioinformatics/补实验/AIE/features/second/test_AAC.csv --type AAC')

os.system('python D:/Study/Bioinformatics/成都培训/git/feature-extraction/iFeature.py --file D:/Study/Bioinformatics/补实验/AIE/fasta/second/train.fasta --out D:/Study/Bioinformatics/补实验/AIE/features/second/train_DPC.csv --type DPC')
os.system('python D:/Study/Bioinformatics/成都培训/git/feature-extraction/iFeature.py --file D:/Study/Bioinformatics/补实验/AIE/fasta/second/test.fasta --out D:/Study/Bioinformatics/补实验/AIE/features/second/test_DPC.csv --type DPC')

os.system('python D:/Study/Bioinformatics/成都培训/git/feature-extraction/codes/CKSAAP.py D:/Study/Bioinformatics/补实验/AIE/fasta/second/train.fasta 3 D:/Study/Bioinformatics/补实验/AIE/features/second/train_CKSAAP.csv')
os.system('python D:/Study/Bioinformatics/成都培训/git/feature-extraction/codes/CKSAAP.py D:/Study/Bioinformatics/补实验/AIE/fasta/second/test.fasta 3 D:/Study/Bioinformatics/补实验/AIE/features/second/test_CKSAAP.csv')

os.system('java -jar D:/Study/Bioinformatics/成都培训/188D(SVMProt)/188D.jar D:\\Study\\Bioinformatics\\补实验\\AIE\\fasta\\second\\train_pos.fasta D:\\Study\\Bioinformatics\\补实验\\AIE\\features\\second\\train_188-bit_pos.arff')
os.system('java -jar D:/Study/Bioinformatics/成都培训/188D(SVMProt)/188D.jar D:\\Study\\Bioinformatics\\补实验\\AIE\\fasta\\second\\train_neg.fasta D:\\Study\\Bioinformatics\\补实验\\AIE\\features\\second\\train_188-bit_neg.arff')
os.system('java -jar D:/Study/Bioinformatics/成都培训/188D(SVMProt)/188D.jar D:\\Study\\Bioinformatics\\补实验\\AIE\\fasta\\second\\test_pos.fasta D:\\Study\\Bioinformatics\\补实验\\AIE\\features\\second\\test_188-bit_pos.arff')
os.system('java -jar D:/Study/Bioinformatics/成都培训/188D(SVMProt)/188D.jar D:\\Study\\Bioinformatics\\补实验\\AIE\\fasta\\second\\test_neg.fasta D:\\Study\\Bioinformatics\\补实验\\AIE\\features\\second\\test_188-bit_neg.arff')

