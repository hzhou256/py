import xlrd
import numpy as np
from scipy import stats


# 导入Excel文件
wb = xlrd.open_workbook(filename='D:\\下载\\HW1.xlsx')
# 打开Sheet1
sheet = wb.sheet_by_name('Sheet1')

name_list = ["沪深300", "大秦铁路", "中国神华", "中国中免", "中煤能源", "大唐发电"]
# 股票的行数（从0开始）
index_list = [1, 7, 11, 41, 42, 50]
data_list = [] # 数据列表

log_List = [] # 对数利润率列表
variance_list = [] # 方差列表


for i in index_list:
    li = sheet.row_values(i)
    code = [0]
    name = li[1]
    data = li[2:]
    # print(name)

    length = len(data)
    # 补上data中缺少的值
    for i in range(length-1):
        if(data[i+1] == ''):
            data[i+1] = data[i]
    data_list.append(data)

    log_Temp = np.zeros(length-1) # 当前股票对数收益率
 
    for i in range(length-1):
        log_Curr = np.log(data[i]) # 当前天的对数值
        log_Next = np.log(data[i+1]) # 后一天的对数值
        log_Temp[i] = log_Next - log_Curr 

    variance = np.var(log_Temp) # 方差

    variance_list.append(variance)
    log_List.append(log_Temp)

print("对数利润率：", log_List)
print("===============================================================================================================")
print("方差：", variance_list)
print("===============================================================================================================")

cov_List = [] # 协方差列表
log_HuShen = log_List[0] #沪深300的对数利润率
for i in range(len(index_list)-1):
    log_Temp = log_List[i+1]
    cov = np.cov(log_HuShen, log_Temp) # 协方差
    cov_List.append(cov[0][1])

print("协方差：", cov_List)
print("===============================================================================================================")

beta_List = [] # beta列表
for i in range(len(cov_List)):
    beta_Temp = cov_List[i] / variance_list[0] # 计算beta
    beta_List.append(beta_Temp)

print("beta：", beta_List)
print("===============================================================================================================")

alpha_List = [] # alpha列表
gamma_M = log_List[0] 
gamma_F = 0.03/243
for i in range(len(log_List)-1):
    gamma = log_List[i+1]
    alpha = gamma - beta_List[i]*(gamma_M - gamma_F) - gamma_F # 计算alpha
    alpha_List.append(alpha)

print("alpha：", alpha_List)

print("===============================================================================================================")
for i in range(len(alpha_List)):
    data = alpha_List[i]
    # print(stats.ttest_1samp(alpha_List[i], 0))
    res =  stats.t.interval(alpha=0.95, df=len(data) - 1, loc=np.mean(data), scale=stats.sem(data)) # 计算置信区间
    print(res)
    if(res[0] < 0 and res[1] > 0): # 判断alpha=0是否在区间内
        print("YES")
    else:
        print("NO")

