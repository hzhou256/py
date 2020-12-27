import pandas as pd
import numpy as np
import math as m
from numpy import *
import matplotlib.pyplot as plt 

#use your local path
path = 'D:/下载/HW1.xlsx'
xlsx = pd.ExcelFile(path)
data = pd.read_excel(xlsx, 'Sheet1')

#data loading,cleaning and preparation
data = data.drop(axis=1,columns='code')
data = data.set_index('name')
data.columns.name = 'time'
data = data.stack().unstack(0)
data = data.fillna(method = 'ffill',axis=0)
data = data.fillna(method = 'bfill',axis=0)

assets = data.columns
date = list(data.index)

date_points = ['20150105','20150701','20160104','20160701','20170103','20170703',
               '20180102','20180702','20190102','20190701','20191231']

def month(day):
    return day[4:6]
    
location_list = [] 
for day in date:
    if not location_list:
        location = date.index(day)
        location_list.append(location)
    elif month(day)!=month(date[location_list[-1]]):
        location = date.index(day)
        location_list.append(location)

month_data = data.loc[data.index[location_list],:]
new_date = list(month_data.index)
month_data_array = np.array(month_data)
rows,columns = month_data_array.shape
month_return = np.zeros((rows-1,columns))
for i in range(columns):
    for j in range(rows-1):
        month_return[j,i] = m.log(month_data_array[j+1,i]/month_data_array[j,i])

month_rf_return = 1.03**(1/12)-1
month_return_withrf = month_return
month_return_withrf[:,0] = month_rf_return

def find_now_month_return(point,new_date,month_return):
    position = new_date.index(point)
    return month_return[:position-1,1:]

def find_A_mu_sigma(now_month_return):
    mu = mat(now_month_return.mean(axis=0)).T
    sigma = mat(np.cov(now_month_return.T))
    one_vector = mat(np.ones(len(mu))).T
    A = mat(np.append(one_vector,mu,axis=1))
    return A,mu,sigma

def cal_B(A,sigma,b1,b2):
    B0 = A.T*sigma.I*A*b2
    B = mat(np.append(b1,B0,axis=1))
    return B

def cal_paras(B):
    aim_vec = mat([1,0.1/12]).T
    result = B.I*aim_vec
    weight0 = result[0]
    lam2 = result[1]
    return float(weight0),float(lam2)

def cal_weight(sigma,A,lam2,b2):
    lam = lam2*b2
    weight = sigma.I*A*lam
    return lam,weight

def cal_final_weight(sigma,A,lam2,b2,B):
    weight0,lam2 = cal_paras(B)
    lam,weight = cal_weight(sigma, A, lam2, b2)
    final_weight = mat(np.append(weight0,weight)).T
    return final_weight

def weight_at_point(point,date,month_return,b1,b2):
    now_month_return = find_now_month_return(point, date, month_return)
    A,mu,sigma = find_A_mu_sigma(now_month_return)
    B = cal_B(A, sigma, b1, b2)
    weight0,lam2 = cal_paras(B)
    lam,weight = cal_weight(sigma, A, lam2, b2)
    final_weight = cal_final_weight(sigma, A, lam2, b2,B)
    return final_weight,mu,sigma

length = len(date_points)-1
height = len(assets)
b1 = mat([1,month_rf_return]).T
b2 = mat([-1*month_rf_return,1]).T
final_weight,mu,sigma= weight_at_point(date_points[0], new_date, month_return, b1,b2)

mu = np.array(mu)
final_weight = np.array(final_weight)
for i in range(1,length):
    point = date_points[i]
    final_weight1,mu1,sigma= weight_at_point(point, new_date, month_return, b1,b2)
    weight1 = np.array(final_weight1)
    mu1 = np.array(mu1)
    final_weight = np.hstack((final_weight,final_weight1))
    mu = np.hstack((mu,mu1))

time_point = []
for point in date_points[:-1]:
    time_point.append(new_date.index(point))
time_point.append(len(new_date)-1)
modified_time = np.array(time_point) - time_point[0]
    
value=3641.5
value_matrix = zeros((time_point[-1]-time_point[0],height))
value_list = []
for i in range(len(modified_time)-1):
    n1 = modified_time[i]
    n2 = modified_time[i+1]    
    for time in range(n1,n2):
        value_list.append(value)
        if time in modified_time:
            for j in range(height):
                value_matrix[time,j]=value_list[-1]*final_weight[j,i]
        else:
            for j in range(height):
                    value_matrix[time,j] = value_matrix[time-1,j]*(1+month_return_withrf[time-1+time_point[0],j])
        value = value_matrix[time,:].sum()
        
hs300 = np.array(month_data.iloc[time_point[0]:,0])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(value_list,'k',label='my own portfolio')
ticks = ax.set_xticks(modified_time)
labels = ax.set_xticklabels(date_points,
                            rotation=30,fontsize='small')
ax.set_title('Value of Portfolio VS Value of HS300')
ax.set_xlabel('time')
ax.set_ylabel('value')
ax.plot(hs300,'r-',label='HS300')
ax.legend(loc='best')
ax.annotate('3641.5',(0,3700))
ax.annotate('3437.9',(55,3500))
    
