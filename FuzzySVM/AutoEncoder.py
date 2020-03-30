import numpy as np
from sklearn import preprocessing
from keras.models import Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt


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
for ds in range(1):
    name_ds = dataset_name[ds]
    print('dataset:', name_ds)
    methods_name = ['188-bit', 'AAC', 'ASDC', 'CKSAAP', 'DPC']
    for it in range(0,1):
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
        encoding_dim = 10

        # this is our input placeholder
        input_img = Input(shape=(188,))

        # 编码层
        encoded = Dense(188, activation='relu')(input_img)
        encoded = Dense(128, activation='relu')(encoded)
        encoded = Dense(64, activation='relu')(encoded)
        encoded = Dense(10, activation='relu')(encoded)
        encoder_output = Dense(encoding_dim)(encoded)

        # 解码层
        decoded = Dense(10, activation='relu')(encoder_output)
        decoded = Dense(64, activation='relu')(decoded)
        decoded = Dense(128, activation='relu')(decoded)
        decoded = Dense(188, activation='tanh')(decoded)

        # 构建自编码模型
        autoencoder = Model(inputs=input_img, outputs=decoded)

        # 构建编码模型
        encoder = Model(inputs=input_img, outputs=encoder_output)

        # compile autoencoder
        autoencoder.compile(optimizer='adam', loss='mse')

        # training
        autoencoder.fit(X_train_scale, X_train_scale, epochs=200, batch_size=64, shuffle=True)
        recon_data = autoencoder.predict(X_train_scale)
        print(X_train_scale)
        print(recon_data)
        
        temp = X_train_scale - recon_data
        recon_error = np.zeros(np.shape(X_train_scale)[0])
        for i in range(np.shape(X_train_scale)[0]):
            recon_error[i] = (np.linalg.norm(temp[i]))**2
        membership = np.exp(-recon_error)
        print(membership)
