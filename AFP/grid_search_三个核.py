import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection
from imblearn.metrics import specificity_score
from sklearn.model_selection import GridSearchCV


dataset_name = ['Antifp_Main', 'Antifp_DS1', 'Antifp_DS2']
for ds in range(3):
    name_ds = dataset_name[ds]
    print('dataset:', name_ds)

    f1 = np.loadtxt('D:/Study/Bioinformatics/AFP/kernel_matrix_3/' + name_ds + '/KM_train_tanimoto/combine_tanimoto_train.csv', delimiter = ',')
    f2 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds + '/train_label.csv', delimiter = ',')

    np.set_printoptions(suppress = True)
    gram = f1
    y_train = f2

    cv = model_selection.StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
    parameters = {'C': np.logspace(-10, 10, base = 10)}
    grid = GridSearchCV(svm.SVC(kernel = 'precomputed', probability = True), parameters, n_jobs = -1, cv = cv)
    grid.fit(gram, y_train)
    grid.best_params_['C']






