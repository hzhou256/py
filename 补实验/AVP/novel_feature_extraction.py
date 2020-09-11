import os
import csv
import pandas as pd


#os.system('python D:/Study/Bioinformatics/成都培训/git/feature-extraction/iFeature.py --file D:/Study/Bioinformatics/补实验/AVP/fasta/train.fasta --out D:/Study/Bioinformatics/补实验/AVP/features/train_AAC.csv --type AAC')
os.system('python D:/Study/Bioinformatics/成都培训/git/feature-extraction/iFeature.py --file D:/Study/Bioinformatics/补实验/AVP/fasta/novel/novel.fasta --out D:/Study/Bioinformatics/补实验/AVP/features/novel/novel_AAC.csv --type AAC')

#os.system('python D:/Study/Bioinformatics/成都培训/git/feature-extraction/iFeature.py --file D:/Study/Bioinformatics/补实验/AVP/fasta/train.fasta --out D:/Study/Bioinformatics/补实验/AVP/features/train_DPC.csv --type DPC')
os.system('python D:/Study/Bioinformatics/成都培训/git/feature-extraction/iFeature.py --file D:/Study/Bioinformatics/补实验/AVP/fasta/novel/novel.fasta --out D:/Study/Bioinformatics/补实验/AVP/features/novel/novel_DPC.csv --type DPC')

#os.system('python D:/Study/Bioinformatics/成都培训/git/feature-extraction/codes/CKSAAP.py D:/Study/Bioinformatics/补实验/AVP/fasta/train.fasta 4 D:/Study/Bioinformatics/补实验/AVP/features/train_CKSAAP.csv')
os.system('python D:/Study/Bioinformatics/成都培训/git/feature-extraction/codes/CKSAAP.py D:/Study/Bioinformatics/补实验/AVP/fasta/novel/novel.fasta 4 D:/Study/Bioinformatics/补实验/AVP/features/novel/novel_CKSAAP.csv')

#os.system('java -jar D:/Study/Bioinformatics/成都培训/188D(SVMProt)/188D.jar D:\\Study\\Bioinformatics\\补实验\\AVP\\fasta\\novel\\novel_pos.fasta D:\\Study\\Bioinformatics\\补实验\\AVP\\features\\novel\\novel_188-bit_pos.arff')
#os.system('java -jar D:/Study/Bioinformatics/成都培训/188D(SVMProt)/188D.jar D:\\Study\\Bioinformatics\\补实验\\AVP\\fasta\\novel\\novel_neg.fasta D:\\Study\\Bioinformatics\\补实验\\AVP\\features\\novel\\novel_188-bit_neg.arff')
#os.system('java -jar D:/Study/Bioinformatics/成都培训/188D(SVMProt)/188D.jar D:\\Study\\Bioinformatics\\补实验\\AVP\\fasta\\test_pos.fasta D:\\Study\\Bioinformatics\\补实验\\AVP\\features\\188\\test_188-bit_pos.arff')
#os.system('java -jar D:/Study/Bioinformatics/成都培训/188D(SVMProt)/188D.jar D:\\Study\\Bioinformatics\\补实验\\AVP\\fasta\\test_neg.fasta D:\\Study\\Bioinformatics\\补实验\\AVP\\features\\188\\test_188-bit_neg.arff')

