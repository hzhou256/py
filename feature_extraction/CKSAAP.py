import os
import csv
import pandas as pd


os.system('python D:/Study/Bioinformatics/成都培训/git/feature-extraction/codes/CKSAAP.py D:/Study/Bioinformatics/补实验/AIE/fasta/train.fasta 5 D:/Study/Bioinformatics/补实验/AIE/features/train_CKSAAP.csv')
os.system('python D:/Study/Bioinformatics/成都培训/git/feature-extraction/codes/CKSAAP.py D:/Study/Bioinformatics/补实验/AIE/fasta/test.fasta 5 D:/Study/Bioinformatics/补实验/AIE/features/test_CKSAAP.csv')