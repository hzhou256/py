import os
import csv
import pandas as pd


#os.system('python D:/Study/Bioinformatics/成都培训/git/feature-extraction/iFeature.py --file D:/Study/Bioinformatics/补实验/AIE/fasta/train.fasta --out D:/Study/Bioinformatics/补实验/AIE/features/train_AAC.csv --type AAC')
#os.system('python D:/Study/Bioinformatics/成都培训/git/feature-extraction/iFeature.py --file D:/Study/Bioinformatics/补实验/AIE/fasta/test.fasta --out D:/Study/Bioinformatics/补实验/AIE/features/test_AAC.csv --type AAC')

#os.system('python D:/Study/Bioinformatics/成都培训/git/feature-extraction/iFeature.py --file D:/Study/Bioinformatics/补实验/AIE/fasta/train.fasta --out D:/Study/Bioinformatics/补实验/AIE/features/train_DPC.csv --type DPC')
#os.system('python D:/Study/Bioinformatics/成都培训/git/feature-extraction/iFeature.py --file D:/Study/Bioinformatics/补实验/AIE/fasta/test.fasta --out D:/Study/Bioinformatics/补实验/AIE/features/test_DPC.csv --type DPC')

#os.system('python D:/Study/Bioinformatics/成都培训/git/feature-extraction/codes/CKSAAP.py E:/Study/Bioinformatics/AMP/fasta/test.fasta 5 E:/Study/Bioinformatics/AMP/features/CKSAAP/test_CKSAAP.csv')

os.system('java -jar D:/Study/Bioinformatics/成都培训/188D(SVMProt)/188D.jar D:\\Study\\Bioinformatics\\补实验\\AIE\\fasta\\temp\\pos_training.txt D:\\Study\\Bioinformatics\\补实验\\AIE\\features\\188\\train_188-bit_pos.arff')
os.system('java -jar D:/Study/Bioinformatics/成都培训/188D(SVMProt)/188D.jar D:\\Study\\Bioinformatics\\补实验\\AIE\\fasta\\temp\\neg_training.txt D:\\Study\\Bioinformatics\\补实验\\AIE\\features\\188\\train_188-bit_neg.arff')
os.system('java -jar D:/Study/Bioinformatics/成都培训/188D(SVMProt)/188D.jar D:\\Study\\Bioinformatics\\补实验\\AIE\\fasta\\temp\\pos_validation.txt D:\\Study\\Bioinformatics\\补实验\\AIE\\features\\188\\test_188-bit_pos.arff')
os.system('java -jar D:/Study/Bioinformatics/成都培训/188D(SVMProt)/188D.jar D:\\Study\\Bioinformatics\\补实验\\AIE\\fasta\\temp\\neg_validation.txt D:\\Study\\Bioinformatics\\补实验\\AIE\\features\\188\\test_188-bit_neg.arff')

