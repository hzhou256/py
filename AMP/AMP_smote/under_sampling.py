import numpy as np
import csv
import pandas as pd
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection
from sklearn.metrics import hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from imblearn.under_sampling import ClusterCentroids


#AMP_multiple = {'antibacterial':1,'anticancer/tumor':2,'antifungal':3,'anti-HIV':4,'antiviral':5}
methods_name = ['188-bit', 'AAC', 'ASDC', 'CKSAAP', 'CTD']
for it in range(5):
    name = methods_name[it]
    data = pd.read_csv('D:/Study/Bioinformatics/AMP_smote/' + name + '/train_' + name + '_smote_2.csv')
    np.set_printoptions(suppress = True)

    X = data.iloc[:, 1:-5]
    y = data.iloc[:, 0]
    groupby_data_orgianl = data.groupby('class').count()
    print(groupby_data_orgianl)

    sampling_strategy_2 = {1:297, 3:297}
  
    model_smote = ClusterCentroids(sampling_strategy = sampling_strategy_2, random_state = 0, n_jobs = -1)
    X_smote_resampled, y_smote_resampled = model_smote.fit_sample(X, y)
    X_smote_resampled = pd.DataFrame(X_smote_resampled, columns = X.columns)
    y_smote_resampled = pd.DataFrame(y_smote_resampled, columns = ['class']) 
    smote_resampled = pd.concat([y_smote_resampled, X_smote_resampled], axis=1) 
    groupby_data_smote = smote_resampled.groupby('class').count()
    print(groupby_data_smote)
    pd.DataFrame.to_csv(smote_resampled, 'D:/Study/Bioinformatics/AMP_smote/' + name + '/train_' + name + '_smote_3.csv', index = None)