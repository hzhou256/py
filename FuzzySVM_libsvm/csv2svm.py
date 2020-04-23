import os


dataset_name = ['Antifp_Main', 'Antifp_DS1', 'Antifp_DS2']
for ds in range(3):
    name_ds = dataset_name[ds]
    print('dataset:', name_ds)
    os.system('python E:/Study/Bioinformatics/FuzzySVM/feature_matrix/CSVtoSVM.py E:/Study/Bioinformatics/FuzzySVM/feature_matrix/'+name_ds+'/188-bit/train_188-bit.csv E:/Study/Bioinformatics/FuzzySVM/feature_matrix/'+name_ds+'/188-bit/train_188-bit.svm')
    os.system('python E:/Study/Bioinformatics/FuzzySVM/feature_matrix/CSVtoSVM.py E:/Study/Bioinformatics/FuzzySVM/feature_matrix/'+name_ds+'/188-bit/test_188-bit.csv E:/Study/Bioinformatics/FuzzySVM/feature_matrix/'+name_ds+'/188-bit/test_188-bit.svm')

    os.system('python E:/Study/Bioinformatics/FuzzySVM/feature_matrix/CSVtoSVM.py E:/Study/Bioinformatics/FuzzySVM/feature_matrix/'+name_ds+'/AAC/train_AAC.csv E:/Study/Bioinformatics/FuzzySVM/feature_matrix/'+name_ds+'/AAC/train_AAC.svm')
    os.system('python E:/Study/Bioinformatics/FuzzySVM/feature_matrix/CSVtoSVM.py E:/Study/Bioinformatics/FuzzySVM/feature_matrix/'+name_ds+'/AAC/test_AAC.csv E:/Study/Bioinformatics/FuzzySVM/feature_matrix/'+name_ds+'/AAC/test_AAC.svm')

    os.system('python E:/Study/Bioinformatics/FuzzySVM/feature_matrix/CSVtoSVM.py E:/Study/Bioinformatics/FuzzySVM/feature_matrix/'+name_ds+'/ASDC/train_ASDC.csv E:/Study/Bioinformatics/FuzzySVM/feature_matrix/'+name_ds+'/ASDC/train_ASDC.svm')
    os.system('python E:/Study/Bioinformatics/FuzzySVM/feature_matrix/CSVtoSVM.py E:/Study/Bioinformatics/FuzzySVM/feature_matrix/'+name_ds+'/ASDC/test_ASDC.csv E:/Study/Bioinformatics/FuzzySVM/feature_matrix/'+name_ds+'/ASDC/test_ASDC.svm')

    os.system('python E:/Study/Bioinformatics/FuzzySVM/feature_matrix/CSVtoSVM.py E:/Study/Bioinformatics/FuzzySVM/feature_matrix/'+name_ds+'/DPC/train_DPC.csv E:/Study/Bioinformatics/FuzzySVM/feature_matrix/'+name_ds+'/DPC/train_DPC.svm')
    os.system('python E:/Study/Bioinformatics/FuzzySVM/feature_matrix/CSVtoSVM.py E:/Study/Bioinformatics/FuzzySVM/feature_matrix/'+name_ds+'/DPC/test_DPC.csv E:/Study/Bioinformatics/FuzzySVM/feature_matrix/'+name_ds+'/DPC/test_DPC.svm')

    os.system('python E:/Study/Bioinformatics/FuzzySVM/feature_matrix/CSVtoSVM.py E:/Study/Bioinformatics/FuzzySVM/feature_matrix/'+name_ds+'/CKSAAP/train_CKSAAP.csv E:/Study/Bioinformatics/FuzzySVM/feature_matrix/'+name_ds+'/CKSAAP/train_CKSAAP.svm')
    os.system('python E:/Study/Bioinformatics/FuzzySVM/feature_matrix/CSVtoSVM.py E:/Study/Bioinformatics/FuzzySVM/feature_matrix/'+name_ds+'/CKSAAP/test_CKSAAP.csv E:/Study/Bioinformatics/FuzzySVM/feature_matrix/'+name_ds+'/CKSAAP/test_CKSAAP.svm')

    os.system('python E:/Study/Bioinformatics/FuzzySVM/feature_matrix/CSVtoSVM.py E:/Study/Bioinformatics/FuzzySVM/feature_matrix/'+name_ds+'/CTD/train_CTD.csv E:/Study/Bioinformatics/FuzzySVM/feature_matrix/'+name_ds+'/CTD/train_CTD.svm')
    os.system('python E:/Study/Bioinformatics/FuzzySVM/feature_matrix/CSVtoSVM.py E:/Study/Bioinformatics/FuzzySVM/feature_matrix/'+name_ds+'/CTD/test_CTD.csv E:/Study/Bioinformatics/FuzzySVM/feature_matrix/'+name_ds+'/CTD/test_CTD.svm')

