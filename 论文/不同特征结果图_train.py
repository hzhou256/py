import xlrd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


df_Main = pd.Series([0.8767,0.8185,0.8476,0.6971,0.8887,0.8502,0.8694,0.7398,0.9067,0.8904,0.8986,0.7974,0.8930,0.8947,0.8938,0.7879,0.8870,0.8819,0.8844,0.7691])
df_DS1 = pd.Series([0.8365,0.8040,0.8202,0.6408,0.8605,0.8373,0.8489,0.6985,0.8956,0.8647,0.8801,0.7610,0.8981,0.8938,0.8960,0.7922,0.8733,0.8879,0.8806,0.7617])
df_DS2 = pd.Series([0.9238,0.9169,0.9204,0.8410,0.9187,0.9152,0.9169,0.8340,0.9409,0.9315,0.9362,0.8727,0.9307,0.9478,0.9392,0.8788,0.9272,0.9358,0.9315,0.8633])
dataset_name = ['Antifp_Main', 'Antifp_DS1', 'Antifp_DS2']
for ds in range(3):
    name_ds = dataset_name[ds]
    print('dataset:', name_ds)


    df = pd.DataFrame(columns = ['Measurement', 'Feature', 'data'])
    df['Measurement'] = pd.Series(['SN','SP','ACC','MCC','SN','SP','ACC','MCC','SN','SP','ACC','MCC','SN','SP','ACC','MCC','SN','SP','ACC','MCC'])
    df['Feature'] = pd.Series(['188-bit','188-bit','188-bit','188-bit','AAC','AAC','AAC','AAC','ASDC','ASDC','ASDC','ASDC','CKSAAP','CKSAAP','CKSAAP','CKSAAP','DPC','DPC','DPC','DPC'])
    if ds == 1:
        df['data'] = df_Main
    elif ds == 2:
        df['data'] = df_DS1
    else:
        df['data'] = df_DS2              

    #print(df)
    sns.set_palette("RdBu")
    sns.catplot(x = "Measurement", y = "data", hue = "Feature", hue_order = ['CKSAAP','ASDC','DPC','AAC','188-bit'], kind = "bar", data = df, legend = False, height = 4, aspect = 1)
    plt.legend(loc = 'upper right', fontsize = 8)
    plt.ylim((0.65, 1))
    plt.xlabel(name_ds + " Training set")
    plt.ylabel("")
    plt.savefig("E:/论文/图表/不同特征结果图/" + name_ds + "_train.png")
    plt.show()