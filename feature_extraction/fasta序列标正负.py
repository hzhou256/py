f2 = open('E:/Study/Bioinformatics/AMP/dataset/test_negative_fix.fasta', 'w')
with open('E:/Study/Bioinformatics/AMP/dataset/test_negative.fasta') as f1:
    line = f1.readlines()
    length = len(line)
    for i in range(length):
        if line[i].startswith('>'):
            f2.write(line[i])
            line_new = line[i+1].strip('\n')
            if ((i+2) < length) and not line[i+2].startswith('>'):
                    line_new = line_new + line[i+2].strip('\n')
            line_new = line_new + '\n'
            f2.write(line_new)


f1.close()
f2.close()

f2 = open('E:/Study/Bioinformatics/AMP/dataset/test_negative_label.fasta', 'w')
with open('E:/Study/Bioinformatics/AMP/dataset/test_negative_fix.fasta') as f1:
    line = f1.readlines()
    length = len(line)
    for i in range(length):
        if line[i].startswith('>'):
            line_new = '>|0' + '\n'
            f2.write(line_new)
        else:
            f2.write(line[i])
f1.close()
f2.close()

f2 = open('E:/Study/Bioinformatics/AMP/dataset/test_positive_fix.fasta', 'w')
with open('E:/Study/Bioinformatics/AMP/dataset/test_positive.fasta') as f1:
    line = f1.readlines()
    length = len(line)
    for i in range(length):
        if line[i].startswith('>'):
            f2.write(line[i])
            line_new = line[i+1].strip('\n')
            if ((i+2) < length) and not line[i+2].startswith('>'):
                    line_new = line_new + line[i+2].strip('\n')
            line_new = line_new + '\n'
            f2.write(line_new)
f1.close()
f2.close()
    
f2 = open('E:/Study/Bioinformatics/AMP/dataset/test_positive_label.fasta', 'w')
with open('E:/Study/Bioinformatics/AMP/dataset/test_positive_fix.fasta') as f1:
    line = f1.readlines()
    length = len(line)
    for i in range(length):
        if line[i].startswith('>'):
            line_new = '>|1' + '\n'
            f2.write(line_new)
        elif line[i] != '\n':
            f2.write(line[i])
f1.close()
f2.close()