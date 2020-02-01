import numpy as np
import csv

def read_file_np(file_path):
    fo = open(file_path, 'r')
    reader = csv.reader(fo, delimiter=",")
    num_cols = len(next(reader))  # Read first line and count columns
    fo.close()
    data = np.loadtxt(file_path, dtype='float', delimiter=",", skiprows=1, usecols=np.arange(1, num_cols))
    return data

def read_label(file_path):
    fo = open(file_path, 'r')
    reader = csv.reader(fo, delimiter=",")
    num_cols = len(next(reader))  # Read first line and count columns
    fo.close()
    label = np.loadtxt(file_path, dtype='float', delimiter=",", usecols=np.arange(0, num_cols))
    return label

def read_file_np1(file_path):
    fo = open(file_path, 'r')
    reader = csv.reader(fo, delimiter=",")
    num_cols = len(next(reader))  # Read first line and count columns
    fo.close()
    data = np.loadtxt(file_path, dtype='float', delimiter=",", usecols=np.arange(0, num_cols))
    return data