f2 = open('E:/Study/Bioinformatics/AMP/dataset/train_negative_fix.fasta', 'w')

with open('E:/Study/Bioinformatics/AMP/dataset/train_negative.fasta') as f1:
    line = f1.readlines()
    length = len(line)
    for i in range(length):
        if line[i].startswith('>'):
            f2.write(line[i])
            line_new = line[i+1].strip('\n')
            if ((i+2) <= length):
                if line[i+2].startswith('>'):
                    line_new = line_new + line[i+2].strip('\n')
            line_new = line_new + '\n'
            f2.write(line_new)

f1.close()
f2.close()