import numpy as np
import csv
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection
from sklearn.metrics import hamming_loss
from imblearn.metrics import specificity_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier

#AMP_multiple = {'antibacterial':1,'anticancer/tumor':2,'antifungal':3,'anti-HIV':4,'antiviral':5}

f1 = np.loadtxt('C:/学习/Bioinformatics/AMP_multiple/kernel_matrix/KM_train_cosine/combine_cosine_train.csv', delimiter = ',')
f3 = np.loadtxt('C:/学习/Bioinformatics/AMP_multiple/kernel_matrix/KM_test_cosine/combine_cosine_test.csv', delimiter = ',')

label = []

with open('C:/学习/Bioinformatics/AMP_multiple/label.txt', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        label.append(line)
mb = MultiLabelBinarizer(classes = ['1','2','3','4','5'])
label_mb = mb.fit_transform(label)

np.set_printoptions(suppress = True)

gram = f1
y_train = label_mb
gram_test = f3

binary_model = svm.SVC(kernel = 'precomputed', probability = True)
multi_model = OneVsRestClassifier(binary_model)
multi_model.fit(gram, y_train)
y_pred = multi_model.predict(gram_test)

print(y_pred)

cv = model_selection.KFold(n_splits = 10, shuffle = True)
ham = metrics.make_scorer(hamming_loss)

scorer = {'ACC':'accuracy', 'hamming_loss':ham, 'AUC':'roc_auc'}
ten_fold = model_selection.cross_validate(multi_model, gram, y_train, cv = cv, scoring = scorer)
print('ACC', np.mean(ten_fold['test_ACC']))
print('Hamming_loss', np.mean(ten_fold['test_hamming_loss']))
print('AUC', np.mean(ten_fold['test_AUC']))