import xlrd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


font = {'size': 12}

df_Main = pd.Series([0.8660,0.8179,0.8419,0.6846,0.8797,0.8454,0.8625,0.7255,0.8866,0.8866,0.8866,0.7732,0.8969,0.9072,0.9021,0.8042,0.9038,0.9003,0.9021,0.8041])
df_DS1 = pd.Series([0.8144,0.8419,0.8282,0.6566,0.8557,0.8488,0.8522,0.7045,0.8763,0.8866,0.8814,0.7629,0.8935,0.9107,0.9021,0.8042,0.8557,0.8935,0.8746,0.7497])
df_DS2 = pd.Series([0.9381,0.9003,0.9192,0.8391,0.9175,0.9038,0.9107,0.8214,0.9175,0.9313,0.9244,0.8489,0.9347,0.9381,0.9364,0.8729,0.9175,0.9519,0.9347,0.8699])
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
    plt.legend(loc = 'upper right', fontsize = 8.5)
    plt.ylim((0.65, 1))
    plt.xlabel(name_ds + " Testing set", font)
    plt.ylabel("ACC", font)  


    plt.tight_layout()
    #plt.savefig("D:/Study/论文/achemso_0825/figure/feature_result/" + name_ds + "_test.png", dpi = 600)
    plt.show()