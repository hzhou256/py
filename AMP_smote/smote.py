import numpy as np
import csv
import pandas as pd
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection
from sklearn.metrics import hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from imblearn.over_sampling import SMOTE

#AMP_multiple = {'antibacterial':1,'anticancer/tumor':2,'antifungal':3,'anti-HIV':4,'antiviral':5}
methods_name = ['188-bit', 'AAC', 'ASDC', 'CKSAAP', 'CTD']
for it in range(5):
    name = methods_name[it]
    data = pd.read_csv('D:/Study/Bioinformatics/AMP_smote/' + name + '/train_' + name + '.csv')
    np.set_printoptions(suppress = True)

    X = data.iloc[:, 1:-5]
    y = data.iloc[:, 0]
    groupby_data_orgianl = data.groupby('class').count()
    print(groupby_data_orgianl)
    max = np.max(groupby_data_orgianl.iloc[:,0])
    avg = int(np.mean(groupby_data_orgianl.iloc[:,0]))

    sampling_strategy_1 = {2:max, 3:max, 4:max, 5:max}
    print(max)
    model_smote = SMOTE(sampling_strategy_1, random_state = 0)
    X_smote_resampled, y_smote_resampled = model_smote.fit_sample(X, y)
    X_smote_resampled = pd.DataFrame(X_smote_resampled, columns = X.columns)
    y_smote_resampled = pd.DataFrame(y_smote_resampled,columns = ['class']) 
    smote_resampled = pd.concat([y_smote_resampled, X_smote_resampled], axis=1) 
    groupby_data_smote = smote_resampled.groupby('class').count()
    print(groupby_data_smote)
    pd.DataFrame.to_csv(smote_resampled, 'D:/Study/Bioinformatics/AMP_smote/' + name + '/train_' + name + '_smote_1.csv', index = None)

    sampling_strategy_2 = {2:avg, 4:avg, 5:avg}
    print(avg)
    model_smote = SMOTE(sampling_strategy_2, n_jobs = -1, random_state = 0)
    X_smote_resampled, y_smote_resampled = model_smote.fit_sample(X, y)
    X_smote_resampled = pd.DataFrame(X_smote_resampled, columns = X.columns)
    y_smote_resampled = pd.DataFrame(y_smote_resampled,columns = ['class']) 
    smote_resampled = pd.concat([y_smote_resampled, X_smote_resampled], axis=1) 
    groupby_data_smote = smote_resampled.groupby('class').count()
    print(groupby_data_smote)
    pd.DataFrame.to_csv(smote_resampled, 'D:/Study/Bioinformatics/AMP_smote/' + name + '/train_' + name + '_smote_2.csv', index = None)