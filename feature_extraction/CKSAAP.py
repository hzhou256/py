import os
from scipy.io import arff
import csv
import pandas as pd


os.system('python E:/Study/成都培训/git/feature-extraction/codes/CKSAAP.py E:/Study/Bioinformatics/DeepAVP/train.fasta 3 E:/Study/Bioinformatics/DeepAVP/CKSAAP/train_CKSAAP.csv')
os.system('python E:/Study/成都培训/git/feature-extraction/codes/CKSAAP.py E:/Study/Bioinformatics/DeepAVP/test.fasta 3 E:/Study/Bioinformatics/DeepAVP/CKSAAP/test_CKSAAP.csv')
