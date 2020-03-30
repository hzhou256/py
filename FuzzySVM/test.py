import sys
import os
path='D:/Program Files/libsvm_weights-3.23/python'
sys.path.append(path)
import numpy as np
from sklearn import preprocessing
from keras.models import Model
from keras.layers import Dense, Input
from keras import regularizers
from sklearn import svm
from sklearn import preprocessing
from svmutil import *
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import collections
import csv


def get_feature(file):
    m = np.shape(file)[0]
    n = np.shape(file)[1]
    data = np.zeros((m, n-1))
    for index in range(m):
        data[index] = file[index][1:]
    return data

def recon_error(x, r):
    return (np.linalg.norm(x - r))**2

dataset_name = ['Antifp_Main', 'Antifp_DS1', 'Antifp_DS2']
for ds in range(1,2):
    name_ds = dataset_name[ds]
    print('dataset:', name_ds)
    methods_name = ['188-bit', 'AAC', 'ASDC', 'CKSAAP', 'DPC']
    for it in range(4,5):
        name = methods_name[it]
        print(name + ':')

        f1 = np.loadtxt('E:/Study/Bioinformatics/FuzzySVM/feature_matrix/' + name_ds +'/' + name + '/train_' + name + '.csv', delimiter = ',', skiprows = 1)
        f2 = np.loadtxt('E:/Study/Bioinformatics/FuzzySVM/feature_matrix/' + name_ds + '/train_label.csv', delimiter = ',')
        X_train = get_feature(f1)
        y_train = f2

        f3 = np.loadtxt('E:/Study/Bioinformatics/FuzzySVM/feature_matrix/' + name_ds +'/' + name + '/test_' + name + '.csv', delimiter = ',', skiprows = 1)
        f4 = np.loadtxt('E:/Study/Bioinformatics/FuzzySVM/feature_matrix/' + name_ds + '/test_label.csv', delimiter = ',')
        X_test = get_feature(f3)
        y_test = f4

        scaler = preprocessing.MinMaxScaler().fit(X_train)
        X_train_scale = scaler.transform(X_train)
        X_test_scale = scaler.transform(X_test)

        # 压缩特征维度
        encoding_dim = 64
        input_dim = np.shape(X_train)[1]
        # this is our input placeholder
        input_img = Input(shape=(input_dim,))

        # 编码层
        encoded = Dense(input_dim, activation='relu', activity_regularizer=regularizers.l1(0.001))(input_img)
        #encoded = Dense(128, activation='relu')(encoded)
        #encoded = Dense(64, activation='relu')(encoded)
        #encoder_output = Dense(encoding_dim)(encoded)

        # 解码层
        #decoded = Dense(64, activation='relu')(encoder_output)
        #decoded = Dense(128, activation='relu')(decoded)
        #decoded = Dense(input_dim, activation='tanh')(decoded)
        decoded = Dense(input_dim, activation='tanh')(encoded)
        # 构建自编码模型
        autoencoder = Model(inputs=input_img, outputs=decoded)

        # 构建编码模型
        #encoder = Model(inputs=input_img, outputs=encoder_output)

        # compile autoencoder
        autoencoder.compile(optimizer='adam', loss='mse')

        # training
        autoencoder.fit(X_train_scale, X_train_scale, epochs=100, batch_size=256, shuffle=True)
        recon_data = autoencoder.predict(X_train_scale)
        
        temp = X_train_scale - recon_data
        recon_error = np.zeros(np.shape(X_train_scale)[0])
        for i in range(np.shape(X_train_scale)[0]):
            recon_error[i] = (np.linalg.norm(temp[i]))**2
        membership = np.exp(-recon_error)
        print(membership)
        #membership = []

        header = np.zeros(np.shape(X_train)[1]+1)
        X_train_scale = np.column_stack((y_train, X_train_scale))
        X_train_scale = np.row_stack((header, X_train_scale))
        X_test_scale = np.column_stack((y_test, X_test_scale))
        X_test_scale = np.row_stack((header, X_test_scale))

        with open('E:/Study/Bioinformatics/FuzzySVM/feature_matrix/' + name_ds +'/' + name + '/train_' + name + '_scale.csv', 'w', newline = '') as csvfile:
            writer = csv.writer(csvfile)
            for row in X_train_scale:
                writer.writerow(row)
            csvfile.close()
        with open('E:/Study/Bioinformatics/FuzzySVM/feature_matrix/' + name_ds +'/' + name + '/test_' + name + '_scale.csv', 'w', newline = '') as csvfile:
            writer = csv.writer(csvfile)
            for row in X_test_scale:
                writer.writerow(row)
            csvfile.close()
        os.system('python E:/Study/Bioinformatics/FuzzySVM/feature_matrix/CSVtoSVM.py E:/Study/Bioinformatics/FuzzySVM/feature_matrix/' + name_ds +'/' + name + '/train_' + name + '_scale.csv E:/Study/Bioinformatics/FuzzySVM/feature_matrix/' + name_ds +'/' + name + '/train_' + name + '_scale.svm')
        os.system('python E:/Study/Bioinformatics/FuzzySVM/feature_matrix/CSVtoSVM.py E:/Study/Bioinformatics/FuzzySVM/feature_matrix/' + name_ds +'/' + name + '/test_' + name + '_scale.csv E:/Study/Bioinformatics/FuzzySVM/feature_matrix/' + name_ds +'/' + name + '/test_' + name + '_scale.svm')
        y_svm, X_svm = svm_read_problem('E:/Study/Bioinformatics/FuzzySVM/feature_matrix/' + name_ds +'/' + name + '/train_' + name + '_scale.svm')
        y_test_svm, X_test_svm = svm_read_problem('E:/Study/Bioinformatics/FuzzySVM/feature_matrix/' + name_ds +'/' + name + '/test_' + name + '_scale.svm')

        def svm_weight_ACC(params, X = X_svm, y = y_svm, W = membership):
            params = {'C': params['C'], 'gamma': params['gamma']}
            prob = svm_problem(W, y, X)
            param = svm_parameter('-t 2 -c '+str(params['C'])+' -g '+str(params['gamma'])+' -v 5')
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
        print('C =', C)
        print('g =', g)
        prob = svm_problem(membership, y_svm, X_svm)
        param = svm_parameter('-t 2 -c '+str(C)+' -g '+str(g))
        m = svm_train(prob, param)
        p_label, p_acc, p_val = svm_predict(y_test_svm, X_test_svm, m)
        print(p_acc[0])
