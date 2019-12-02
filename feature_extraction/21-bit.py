import numpy as np
import csv


#Charge = [['A','C','F','G','H','I','L','M','N','P','Q','S','T','V','W','Y'], ['D', 'E'], ['K', 'R']]
#Hydrophobicity = [['C','F','I','L','M','V','W'], ['A','G','H','P','S','T','Y'], ['D','E','K','N','Q','R']]
#Vander_Waals = [['A','C','D','G','P','S','T'], ['E','I','L','N','Q','V'], ['F','H','K','M','R','W','Y']]
#Polarity = [['C','F','I','L','M','V','W','Y'], ['A','G','P','S','T'], ['D','E','H','K','N','Q','R']]
#Polariizability = [['A','D','G','S','T'], ['C','E','I','L','N','P','Q','V'], ['F','H','K','M','R','W','Y']]
#SS = [['D','G','N','P','S'], ['A','E','H','K','L','M','Q','R'], ['C','F','I','T','V','W','Y']]
#SA = [['A','C','F','G','I','L','V','W'], ['H','M','P','S','T','Y'], ['D','E','K','N','R','Q']]
AA = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
Total = [['A','C','F','G','H','I','L','M','N','P','Q','S','T','V','W','Y'], ['D', 'E'], ['K', 'R'],
    ['C','F','I','L','M','V','W'], ['A','G','H','P','S','T','Y'], ['D','E','K','N','Q','R'],
    ['A','C','D','G','P','S','T'], ['E','I','L','N','Q','V'], ['F','H','K','M','R','W','Y'],
    ['C','F','I','L','M','V','W','Y'], ['A','G','P','S','T'], ['D','E','H','K','N','Q','R'],
    ['A','D','G','S','T'], ['C','E','I','L','N','P','Q','V'], ['F','H','K','M','R','W','Y'],
    ['D','G','N','P','S'], ['A','E','H','K','L','M','Q','R'], ['C','F','I','T','V','W','Y'],
    ['A','C','F','G','I','L','V','W'], ['H','M','P','S','T','Y'], ['D','E','K','N','R','Q']]




N = np.zeros((400,105))
def get_vec(seq, N, index):
    cnt = 0
    for x in range(5):
        for  i in range(21):
            if seq[x] in Total[i]:
                N[index][cnt] = 1
            else:
                N[index][cnt] = 0
            cnt = cnt + 1


f1 = open('D:/Study/生物信息学/Kernel_PCA/群体反应信号肽/datasets/21-bit/C端(右).fasta')
line = f1.readlines()
index = 0
for line_list in line:
    if not line_list.startswith('>'):
        line_new = line_list.strip('/n')
        get_vec(line_new, N, index)
        index = index + 1



with open('D:/Study/生物信息学/Kernel_PCA/群体反应信号肽/datasets/21-bit/C端(右).csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in N:
        writer.writerow(row)

f1.close()
csvfile.close()