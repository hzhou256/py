import numpy as np
from sklearn.metrics import hamming_loss
import operator


y_true = np.loadtxt('D:/Study/Bioinformatics/AMP_smote/multiple_label_1.csv', delimiter = ',')
y_pred = np.loadtxt('D:/Study/Bioinformatics/AMP_smote/kernel_matrix_1/KM_train_cosine/y_predict.csv', delimiter = ',')

def recall(y_pred, y_true):
    N = np.shape(y_true)[0]
    M = np.shape(y_true)[1]
    Sum = 0
    for i in range(N):
        L_true = y_true[i]
        L_pred = y_pred[i]
        count = 0
        for j in range(M):
            if L_pred[j] == 1 and L_true[j] == 1:
                count = count + 1
        denominator = count
        numerator = np.count_nonzero(L_true)
        temp = denominator / numerator
        Sum = Sum + temp
    result = Sum / N
    return result

def precision(y_pred, y_true):
    N = np.shape(y_true)[0]
    M = np.shape(y_true)[1]
    Sum = 0
    for i in range(N):
        L_true = y_true[i]
        L_pred = y_pred[i]
        count = 0
        for j in range(M):
            if L_pred[j] == 1 and L_true[j] == 1:
                count = count + 1
        denominator = count
        numerator = np.count_nonzero(L_true)
        if numerator == 0:
            temp = 0
        else:
            temp = denominator / numerator
        Sum = Sum + temp
    result = Sum / N
    return result

def accuracy(y_pred, y_true):
    N = np.shape(y_true)[0]
    M = np.shape(y_true)[1]
    Sum = 0
    for i in range(N):
        L_true = y_true[i]
        L_pred = y_pred[i]
        count = 0
        for j in range(M):
            if L_pred[j] == 1 and L_true[j] == 1:
                count = count + 1
        denominator = count
        cnt = 0
        for k in range(M):
            if L_pred[k] == 1 or L_true[k] == 1:
                cnt = cnt + 1
        numerator = cnt
        temp = denominator / numerator
        Sum = Sum + temp
    result = Sum / N
    return result

def subset_accuracy(y_pred, y_true):
    N = np.shape(y_true)[0]
    Sum = 0
    for i in range(N):
        L_true = y_true[i]
        L_pred = y_pred[i]
        temp = 0
        if all(operator.eq(L_pred, L_true)) == True:
            temp = 1
        else:
            temp = 0
        Sum = Sum + temp
    result = Sum / N
    return result

Recall = recall(y_pred, y_true)
Hamming_loss = hamming_loss(y_true, y_pred)
Precision = precision(y_pred, y_true)
ACC = accuracy(y_pred, y_true)
Subset_accuracy = subset_accuracy(y_pred, y_true)

print('Hamming_loss =', Hamming_loss)
print('Accuracy =', ACC)
print('Precision =', Precision)
print('Recall =', Recall)
print('Subset_accuracy =', Subset_accuracy)
