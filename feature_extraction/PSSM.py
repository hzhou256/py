import os


dataset_name = ['Antifp_Main', 'Antifp_DS1', 'Antifp_DS2']
for ds in range(3):
    name_ds = dataset_name[ds]
    os.system('python D:/Study/Bioinformatics/成都培训/git/feature-extraction/iFeature.py --file D:/Study/Bioinformatics/AFP/datasets/' + name_ds +'/train.fasta --out D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds +'/PSSM/train_PSSM.txt --type PSSM')
    os.system('python D:/Study/Bioinformatics/成都培训/git/feature-extraction/iFeature.py --file D:/Study/Bioinformatics/AFP/datasets/' + name_ds +'/test.fasta --out D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds +'/PSSM/test_PSSM.txt --type PSSM')
