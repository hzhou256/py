import numpy as np
import membership, Fuzzy_SVM
from sklearn import metrics
from imblearn.metrics import specificity_score
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split, StratifiedKFold


def split(X, y): 
    '''
    y = {-1, 1}
    '''
    n_pos, n_neg = 0, 0
    for y_i in y:
        if y_i == 1:
            n_pos += 1
        elif y_i == -1:
            n_neg += 1
    n = np.shape(X)[1]
    X_pos = np.zeros((n_pos, n)) 
    X_neg = np.zeros((n_neg, n)) 
    j, k = 0, 0
    for i in range(n_pos + n_neg):
        if y[i] == -1:
            X_neg[j] = X[i]
            j = j + 1
        else:
            X_pos[k] = X[i]
            k = k + 1
    y = np.zeros(n_neg + n_pos)
    for i in range(n_neg):
        y[i] = -1
    for i in range(n_neg, n_neg + n_pos):
        y[i] = 1
    X = np.row_stack((X_neg, X_pos))
    return X, y

dataset = ['australian', 'heart', 'sonar']
name = dataset[2]

f1 = np.loadtxt('E:/Study/Bioinformatics/UCI/' + name + '/data.csv', delimiter = ',')
X = f1[:, 0:-1]
y = f1[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_train, y_train = split(X_train, y_train)

C = 2
gamma = 1

clf = Fuzzy_SVM.FSVM_Classifier(C = C, gamma = gamma, membership = 'OCSVM')
cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
five_fold = cross_validate(clf, X_train, y_train, cv = cv, scoring = 'accuracy')
mean_ACC = np.mean(five_fold['test_score'])

print('five fold:')
print(mean_ACC)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)
ACC = metrics.accuracy_score(y_test, y_pred)
AUC = metrics.roc_auc_score(y_test, clf.decision_function(X_test))

print('Testing set:')
print(ACC)
print(AUC)



