import numpy as np


Antifp_Main = {
'188-bit': 0.04591392963551152, 
'AAC': 0.1572197931045989, 
'ASDC': 0.1510183870821174, 
'CKSAAP': 0.11551720749843453, 
'CTD': 0.01811801169658544, 
'DPC': 0.10640206775732973
}
Antifp_DS1 = {
'188-bit': 0.03894170941489469, 
'AAC': 0.07670017309350463, 
'ASDC': 0.0855769081462958, 
'CKSAAP': 0.08588399768350126, 
'CTD': 0.019735160849863168, 
'DPC': 0.0789350218050075
}
Antifp_DS2 ={
'188-bit': 0.09806313031514946, 
'AAC': 0.3329549194148061, 
'ASDC': 0.31035689826792273, 
'CKSAAP': 0.20478317402527116, 
'CTD': 0.044514108755554734, 
'DPC': 0.18527440010598356
}


print('Antifp_DS1')
sum = Antifp_DS1['AAC']*0 + Antifp_DS1['CKSAAP'] + Antifp_DS1['ASDC'] + Antifp_DS1['DPC']*0 + Antifp_DS1['188-bit']*0 + Antifp_DS1['CTD']*0
#weight0 = Antifp_DS1['188-bit'] / sum
#weight1 = Antifp_DS1['AAC'] / sum
weight2 = Antifp_DS1['ASDC'] / sum
weight3 = Antifp_DS1['CKSAAP'] / sum
#weight4 = Antifp_DS1['CTD'] / sum
#weight5 = Antifp_DS1['DPC'] / sum

#print(weight0)
#print(weight1)
print(weight2)
print(weight3)
#print(weight4)
#print(weight5)


print('Antifp_DS2')
sum = Antifp_DS2['CKSAAP'] + Antifp_DS2['ASDC'] + Antifp_DS2['DPC']*0 + Antifp_DS2['AAC']*0 + Antifp_DS2['188-bit']*0 + Antifp_DS2['CTD']*0
#weight0 = Antifp_DS2['188-bit'] / sum
#weight1 = Antifp_DS2['AAC'] / sum
weight2 = Antifp_DS2['ASDC'] / sum
weight3 = Antifp_DS2['CKSAAP'] / sum
#weight4 = Antifp_DS2['CTD'] / sum
#weight5 = Antifp_DS2['DPC'] / sum

#print(weight0)
#print(weight1)
print(weight2)
print(weight3)
#print(weight4)
#print(weight5)


print('Antifp_Main')
sum = Antifp_Main['CKSAAP'] + Antifp_Main['ASDC'] + Antifp_Main['DPC']*0 + Antifp_Main['AAC']*0 + Antifp_Main['188-bit']*0 + Antifp_Main['CTD']*0
#weight0 = Antifp_Main['188-bit'] / sum
#weight1 = Antifp_Main['AAC'] / sum
weight2 = Antifp_Main['ASDC'] / sum
weight3 = Antifp_Main['CKSAAP'] / sum
#weight4 = Antifp_Main['CTD'] / sum
#weight5 = Antifp_Main['DPC'] / sum

#print(weight0)
#print(weight1)
print(weight2)
print(weight3)
#print(weight4)
#print(weight5)


