import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection
from imblearn.metrics import specificity_score

c_list = [175.751062485479, 4.09491506238041, 1.59985871960605]
dataset_name = ['Antifp_Main', 'Antifp_DS1', 'Antifp_DS2']
for ds in range(3):
    name_ds = dataset_name[ds]
    print('dataset:', name_ds)

    f1 = np.loadtxt('D:/Study/Bioinformatics/AFP/kernel_matrix_3/' + name_ds + '/KM_train_tanimoto/combine_tanimoto_train.csv', delimiter = ',')
    f2 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds + '/train_label.csv', delimiter = ',')
    f3 = np.loadtxt('D:/Study/Bioinformatics/AFP/kernel_matrix_3/' + name_ds + '/KM_test_tanimoto/combine_tanimoto_test.csv', delimiter = ',')
    f4 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds + '/test_label.csv', delimiter = ',')

    np.set_printoptions(suppress = True)
    gram = f1
    y_train = f2
    gram_test = f3
    y_test = f4

    clf = svm.SVC(C = c_list[ds], kernel = 'precomputed', probability = True)
    clf.fit(gram, y_train)

    y_pred = clf.predict(gram_test)

    ACC = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    sensitivity = metrics.recall_score(y_test, y_pred)
    specificity = specificity_score(y_test, y_pred)
    AUC = metrics.roc_auc_score(y_test, clf.decision_function(gram_test))
    MCC = metrics.matthews_corrcoef(y_test, y_pred)

    #print('precision =', round(precision, 3))
    print('SN =', sensitivity)
    print('SP =', specificity)
    print('ACC =', ACC)
    print('MCC =', MCC)
    print('AUC =', AUC)