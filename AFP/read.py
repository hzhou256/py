import csv
import numpy as np
import pandas as pd

def read_file_np(file_path):
    fo = open(file_path, 'r')
    reader = csv.reader(fo, delimiter=",")
    num_cols = len(next(reader))  # Read first line and count columns
    fo.close()
    data = np.loadtxt(file_path, dtype="float", delimiter=",", skiprows=1, usecols=np.arange(1, num_cols-1))
    regression = np.loadtxt(file_path, dtype="float", delimiter=",", skiprows=1, usecols=(num_cols-1,))
    return data,regression

def read_file_np1(file_path):
    fo = open(file_path, 'r')
    reader = csv.reader(fo, delimiter=",")
    num_cols = len(next(reader))  # Read first line and count columns
    fo.close()
    kernel = np.loadtxt(file_path, dtype="float", delimiter=",", skiprows=0, usecols=np.arange(0, num_cols))
    return kernel

def read_file_np2(file_path):
    fo = open(file_path, 'r')
    reader = csv.reader(fo, delimiter=",")
    num_cols = len(next(reader))  # Read first line and count columns
    fo.close()
    data = np.loadtxt(file_path, dtype="float", delimiter=",", usecols=np.arange(0, num_cols - 1))
    regression = np.loadtxt(file_path, dtype="float", delimiter=",", usecols=(num_cols - 1,))
    return data, regression
def read_file_pd(file_path):

    # print(data.shape)
    # print(data)
    # data1 = data[list(range(1,data.shape[1]-1))]
    # data2 = data[data.shape[1]]
    # return data1,data2
    return 0

# x,y = read_file_np("BoLA-AW10.csv")
# print(x.shape,y.shape)
