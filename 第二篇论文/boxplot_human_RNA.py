import seaborn as sns
import xlrd
import pandas as pd
import matplotlib.pyplot as plt


wb = xlrd.open_workbook(filename='D:\\Study\\Bioinformatics\\王浩\\data and code\\matlab\\result\\human_RNA_result.xlsx')

sheet = wb.sheet_by_name('boxplot')


dataset = sheet.col_values(0)

AP = sheet.col_values(1)
AP_type = sheet.col_values(2)

Coverage = sheet.col_values(3)
Coverage_type = sheet.col_values(4)

One = sheet.col_values(5)
One_type = sheet.col_values(6)

Rloss = sheet.col_values(7)
Rloss_type = sheet.col_values(8)

Hloss = sheet.col_values(9)
Hloss_type = sheet.col_values(10)


dataset_list = pd.Series(data = dataset[1:], name = dataset[0])

AP_list = pd.Series(data = AP[1:])
AP_type_list = pd.Series(data = AP_type[1:])

Coverage_list = pd.Series(data = Coverage[1:])
Coverage_type_list = pd.Series(data = Coverage_type[1:])

One_list = pd.Series(data = One[1:])
One_type_list = pd.Series(data = One_type[1:])

Rloss_list = pd.Series(data = Rloss[1:])
Rloss_type_list = pd.Series(data = Rloss_type[1:])

Hloss_list = pd.Series(data = Hloss[1:])
Hloss_type_list = pd.Series(data = Hloss_type[1:])


fig,axes = plt.subplots(1,5,sharey=False,figsize=(15,7))


sns.set()

sns.boxplot(x=AP_type_list, y=AP_list, hue=dataset_list, ax=axes[0])
sns.boxplot(x=Coverage_type_list, y=Coverage_list, hue=dataset_list, ax=axes[1])
sns.boxplot(x=One_type_list, y=One_list, hue=dataset_list, ax=axes[2])
sns.boxplot(x=Rloss_type_list, y=Rloss_list, hue=dataset_list, ax=axes[3])
sns.boxplot(x=Hloss_type_list, y=Hloss_list, hue=dataset_list, ax=axes[4])

plt.tight_layout()
plt.savefig("D:\\Study\\论文\\achemso_mlghknn\\figure\\boxplot\\human_RNA.png", dpi = 600)
plt.show()