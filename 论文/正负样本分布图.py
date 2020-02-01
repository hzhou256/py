import xlrd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset_name = ['Antifp_Main', 'Antifp_DS1', 'Antifp_DS2']
for ds in range(3):
    name_ds = dataset_name[ds]
    print('dataset:', name_ds)

    train_file = "D:/Study/Bioinformatics/AFP/datasets/" + name_ds + "/main.xlsx"
    wb = xlrd.open_workbook(filename = train_file)
    sheet1 = wb.sheet_by_name('positive')
    sheet2 = wb.sheet_by_name('negative')

    cols_positive = sheet1.col_values(1)
    cols_negative = sheet2.col_values(1)
    n_seq_pos = sheet1.nrows
    n_seq_neg = sheet2.nrows
    temp_pos = np.zeros(n_seq_pos)
    temp_neg = np.zeros(n_seq_neg)

    for i in range(n_seq_pos):
        l = len(cols_positive[i])
        temp_pos[i] = l
    for j in range(n_seq_neg):
        l = len(cols_negative[j])
        temp_neg[j] = l

    plt.figure(figsize = [6.4*2, 4.8])
    ax_train = plt.subplot(121)
    sns.set_palette("hls")
    sns.distplot(temp_pos, color="royalblue", kde=True, label = 'Positive')
    sns.distplot(temp_neg, color="darkorange", kde=True, label = 'Negative')

    plt.title("Length Distribution (Training set)")
    plt.xlabel("Length")
    plt.ylabel("Rate")
    plt.legend()

    test_file = "D:/Study/Bioinformatics/AFP/datasets/" + name_ds + "//validation.xlsx"
    wb = xlrd.open_workbook(filename = test_file)
    sheet1 = wb.sheet_by_name('positive')
    sheet2 = wb.sheet_by_name('negative')

    cols_positive = sheet1.col_values(1)
    cols_negative = sheet2.col_values(1)
    n_seq_pos = sheet1.nrows
    n_seq_neg = sheet2.nrows
    temp_pos = np.zeros(n_seq_pos)
    temp_neg = np.zeros(n_seq_neg)

    for i in range(n_seq_pos):
        l = len(cols_positive[i])
        temp_pos[i] = l
    for j in range(n_seq_neg):
        l = len(cols_negative[j])
        temp_neg[j] = l

    ax_test = plt.subplot(122)
    sns.set_palette("hls")
    sns.distplot(temp_pos, color="royalblue", kde=True, label = 'Positive')
    sns.distplot(temp_neg, color="darkorange", kde=True, label = 'Negative')

    plt.title("Length Distribution (Testing set)")
    plt.xlabel("Length")
    plt.ylabel("Rate")
    plt.legend()

    plt.savefig("D:/论文/图表/正负样本长度分布图/" + name_ds + ".png")
    plt.show()