f2 = open('D:/Study/生物信息学/Kernel_PCA/群体反应信号肽/datasets/Binary/右端.fasta', 'w')

with open('D:/Study/生物信息学/Kernel_PCA/群体反应信号肽/datasets/datasets.fasta') as f1:
    line = f1.readlines()
    for line_list in line:
        if not line_list.startswith('>'):
            l = len(line_list)
            line_new =line_list[l-6:l-1]
            line_new = line_new + '\n'
            f2.write(line_new)
        else:
            f2.write(line_list)

f1.close()
f2.close()