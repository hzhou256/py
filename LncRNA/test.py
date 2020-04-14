f2 = open('E:/Study/Bioinformatics/RNA/ribosome.csv', 'w')

with open('E:/Study/Bioinformatics/RNA/ribosome.txt') as f1:
    line = f1.readlines()
    length = len(line)
    for i in range(length):
        if not line[i].startswith('>'):
            f2.write(line[i])

f1.close()
f2.close()