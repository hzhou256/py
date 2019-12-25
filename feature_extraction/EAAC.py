import os


dataset_name = ['Antifp_Main', 'Antifp_DS1', 'Antifp_DS2']
for ds in range(3):
    name_ds = dataset_name[ds]
    os.system('python D:/Study/Bioinformatics/成都培训/git/feature-extraction/codes/EAAC.py D:/Study/Bioinformatics/AFP/datasets/' + name_ds +'/train.fasta 4 D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds +'/EAAC/train_EAAC.csv')
    os.system('python D:/Study/Bioinformatics/成都培训/git/feature-extraction/codes/EAAC.py D:/Study/Bioinformatics/AFP/datasets/' + name_ds +'/test.fasta 4 D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds +'/EAAC/test_EAAC.csv')
