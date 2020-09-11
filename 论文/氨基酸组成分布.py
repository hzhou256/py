import xlrd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


AA = {'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8, 'L':9, 
'M':10, 'N':11, 'P':12, 'Q':13, 'R':14, 'S':15, 'T':16, 'V':17, 'W':18, 'Y':19}

dataset_name = ['Antifp_Main', 'Antifp_DS1', 'Antifp_DS2']
for ds in range(0, 1):
    name_ds = dataset_name[ds]
    print('dataset:', name_ds)

    train_file = "D:/Study/Bioinformatics/AFP/datasets/" + name_ds + "/" + name_ds + ".xlsx"
    wb = xlrd.open_workbook(filename = train_file)
    sheet1 = wb.sheet_by_name('positive')
    sheet2 = wb.sheet_by_name('negative')

    cols_positive = sheet1.col_values(1)
    cols_negative = sheet2.col_values(1)
    n_seq_pos = sheet1.nrows
    n_seq_neg = sheet2.nrows
    
    sum_pos = 0
    sum_neg = 0
    temp_pos = np.zeros((20, 3))
    temp_neg = np.zeros((20, 3))

    for i in range(n_seq_pos):
        sum_pos = sum_pos + len(cols_positive[i])
        for x in cols_positive[i]:
            if x == 'X':
                continue
            else:
                temp_pos[AA[x]][0] = temp_pos[AA[x]][0] + 1
    for j in range(n_seq_neg):
        sum_neg = sum_neg + len(cols_negative[j])
        for x in cols_negative[j]:
            if x == 'X':
                continue
            else:
                temp_neg[AA[x]][0] = temp_neg[AA[x]][0] + 1
    for i in range(20):
        temp_pos[i][0] = temp_pos[i][0] / sum_pos
        temp_neg[i][0] = temp_neg[i][0] / sum_neg

    result_pos = pd.DataFrame(temp_pos, columns = ['Composition', 'Type', 'Amino acid'])
    result_neg = pd.DataFrame(temp_neg, columns = ['Composition', 'Type', 'Amino acid'])
    result_pos['Type'] = pd.Series(['AFP', 'AFP', 'AFP', 'AFP', 'AFP', 'AFP', 'AFP', 'AFP', 'AFP', 'AFP', 'AFP', 'AFP', 'AFP', 'AFP', 'AFP', 'AFP', 'AFP', 'AFP', 'AFP', 'AFP'])
    result_neg['Type'] = pd.Series(['Non_AFP', 'Non_AFP','Non_AFP','Non_AFP','Non_AFP','Non_AFP','Non_AFP','Non_AFP','Non_AFP','Non_AFP','Non_AFP','Non_AFP','Non_AFP','Non_AFP','Non_AFP','Non_AFP','Non_AFP','Non_AFP','Non_AFP','Non_AFP'])
    result_pos['Amino acid'] = pd.Series(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])
    result_neg['Amino acid'] = pd.Series(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])
    result = pd.concat([result_pos, result_neg], ignore_index = True)

    t, p = stats.ttest_ind(result_pos['Composition'], result_neg['Composition'])
    #print(p)
    sns.catplot(x = "Amino acid", y = "Composition", hue = "Type", kind = "bar", data = result, legend = False, height = 4.8, aspect = 2)
    plt.legend()
    #plt.savefig("D:/论文/图表/氨基酸组成分布图/" + name_ds + ".png")
    plt.show()