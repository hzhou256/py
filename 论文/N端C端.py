dataset_name = ['Antifp_Main', 'Antifp_DS1', 'Antifp_DS2']
for ds in range(3):
    name_ds = dataset_name[ds]
    print('dataset:', name_ds)

    #C端4个残基
    c_file = open('D:/论文/图表/氨基酸logo图/' + name_ds + '/C_positive.fasta', 'w')

    with open('D:/Study/Bioinformatics/AFP/datasets/' + name_ds + '/positive.fasta') as pos_file:
        line = pos_file.readlines()
        for line_list in line:
            if not line_list.startswith('>'):
                l = len(line_list)
                line_new =line_list[l-5:l-1]
                line_new = line_new + '\n'
                c_file.write(line_new)
            else:
                c_file.write(line_list)

    pos_file.close()
    c_file.close()

    c_file = open('D:/论文/图表/氨基酸logo图/' + name_ds + '/C_negative.fasta', 'w')

    with open('D:/Study/Bioinformatics/AFP/datasets/' + name_ds + '/negative.fasta') as neg_file:
        line = neg_file.readlines()
        for line_list in line:
            if not line_list.startswith('>'):
                l = len(line_list)
                line_new =line_list[l-5:l-1]
                line_new = line_new + '\n'
                c_file.write(line_new)
            else:
                c_file.write(line_list)

    neg_file.close()
    c_file.close()

    #N端四个残基
    N_file = open('D:/论文/图表/氨基酸logo图/' + name_ds + '/N_positive.fasta', 'w')

    with open('D:/Study/Bioinformatics/AFP/datasets/' + name_ds + '/positive.fasta') as pos_file:
        line = pos_file.readlines()
        for line_list in line:
            if not line_list.startswith('>'):
                l = len(line_list)
                line_new =line_list[0:4]
                line_new = line_new + '\n'
                N_file.write(line_new)
            else:
                N_file.write(line_list)

    pos_file.close()
    N_file.close()

    N_file = open('D:/论文/图表/氨基酸logo图/' + name_ds + '/N_negative.fasta', 'w')

    with open('D:/Study/Bioinformatics/AFP/datasets/' + name_ds + '/negative.fasta') as neg_file:
        line = neg_file.readlines()
        for line_list in line:
            if not line_list.startswith('>'):
                l = len(line_list)
                line_new =line_list[0:4]
                line_new = line_new + '\n'
                N_file.write(line_new)
            else:
                N_file.write(line_list)

    neg_file.close()
    N_file.close()