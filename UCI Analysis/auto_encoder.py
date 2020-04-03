import sys
import os
import csv
path='D:/Program Files/libsvm_weights-3.23/python'
sys.path.append(path)
import numpy as np
from sklearn import preprocessing
from keras.models import Model
from keras.layers import Dense, Input
from keras import regularizers
from sklearn import preprocessing
from svmutil import *
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.metrics import specificity_score


def recon_error(x, r):
    return (np.linalg.norm(x - r))**2

def get_feature(file):
    m = np.shape(file)[0]
    n = np.shape(file)[1]
    data = np.zeros((m, n-1))
    for index in range(m):
        data[index] = file[index][1:]
    return data


dataset = ['ionosphere', 'german']
name = dataset[1]

f1 = np.loadtxt('E:/Study/Bioinformatics/UCI/' + name + '/X_train.csv', delimiter = ',', skiprows = 1)
X_train = get_feature(f1)
y_train = f1[:, 0]

f2 = np.loadtxt('E:/Study/Bioinformatics/UCI/' + name + '/X_test.csv', delimiter = ',', skiprows = 1)
X_test = get_feature(f2)
y_test = f2[:, 0]


y_svm, X_svm = svm_read_problem('E:/Study/Bioinformatics/UCI/' + name + '/X_train.svm')
y_test_svm, X_test_svm = svm_read_problem('E:/Study/Bioinformatics/UCI/' + name + '/X_test.svm')

# 压缩特征维度
encoding_dim = 30
input_dim = np.shape(X_train)[1]
# this is our input placeholder
input_img = Input(shape=(input_dim,))

# 编码层
encoded = Dense(input_dim, activation='relu', activity_regularizer=regularizers.l1(0.01))(input_img)
encoder_output = Dense(encoding_dim)(encoded)
# 解码层
decoded = Dense(input_dim, activation='tanh')(encoder_output)

# 构建自编码模型
autoencoder = Model(inputs=input_img, outputs=decoded)

# 构建编码模型
#encoder = Model(inputs=input_img, outputs=encoder_output)

# compile autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

# training
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, shuffle=True)
recon_data = autoencoder.predict(X_train)

temp = X_train - recon_data
recon_error = np.zeros((np.shape(X_train)[0], 1))
for i in range(np.shape(X_train)[0]):
    recon_error[i] = (np.linalg.norm(temp[i]))**2
scaler = preprocessing.MinMaxScaler().fit(recon_error)
recon_error_scale = scaler.transform(recon_error)
print(recon_error_scale)
membership = 1 - recon_error_scale
#membership = np.exp(-recon_error)
print(membership)
#membership = []


def svm_weight_ACC(params, X = X_svm, y = y_svm, W = membership):
    params = {'C': params['C'], 'gamma': params['gamma']}
    prob = svm_problem(W, y, X)
    param = svm_parameter('-t 2 -c '+str(params['C'])+' -g '+str(params['gamma'])+' -v 10')
    score = svm_train(prob, param)
    return -score
space = {'C': hp.loguniform('C', low = np.log(1e-7) , high = np.log(1e3)), 'gamma': hp.loguniform('gamma', low = np.log(1e-7) , high = np.log(1e5))}
trials = Trials()
best = fmin(fn = svm_weight_ACC, 
        space = space,
        algo = tpe.suggest, 
        max_evals = 100, 
        trials = trials,
        )
C = best['C']
g = best['gamma']
prob = svm_problem(membership, y_svm, X_svm)
param = svm_parameter('-t 2 -c '+str(C)+' -g '+str(g) + ' -b 1')
m = svm_train(prob, param)
p_label, p_acc, p_val = svm_predict(y_test_svm, X_test_svm, m, '-b 1')
y_prob = np.reshape([p_val[i][0] for i in range(np.shape(p_val)[0])], (np.shape(p_val)[0], 1))
ACC = metrics.accuracy_score(y_test, p_label)
precision = metrics.precision_score(y_test, p_label)
sensitivity = metrics.recall_score(y_test, p_label)
specificity = specificity_score(y_test, p_label)
AUC = metrics.roc_auc_score(y_test, y_prob)
MCC = metrics.matthews_corrcoef(y_test, p_label)

print('C =', C)
print('g =', g)

print('SN =', sensitivity)
print('SP =', specificity)
print('ACC =', p_acc[0])
print('MCC =', MCC)
print('AUC =', AUC)
