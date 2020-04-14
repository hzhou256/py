import os
import time
import numpy as np
import pandas as pd

import keras.layers.core as core
import keras.layers.convolutional as conv
import keras.models as models
import keras.utils.np_utils as kutils
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import merge,BatchNormalization
from keras.layers import pooling
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint,Callback
from keras.layers import Dense, Dropout, Activation, Flatten, Input, MaxPooling1D
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2
from attention import Attention,myFlatten
import copy

def DeepCleave_CNN(output_model_name, trainX, trainY,valX=None, valY=None,
             batch_size=1024, 
             epochs=500,
             n_earlystop=None,n_transfer_layer=1,background_weights=None,for_transfer=False,compiletimes=0,
             compilemodels=None,predict=False):
    input_row     = trainX.shape[1]
    input_col     = trainX.shape[2]

    trainX_t=trainX;
    valX_t=valX;
    checkpoint = ModelCheckpoint(filepath=output_model_name,monitor='val_acc', verbose=1,mode='max' ,save_best_only='True')
    
    if(n_earlystop is not None): 
        early_stopping = EarlyStopping(monitor='val_acc', patience=n_earlystop)
        epochs=10000;#set to a very big value since earlystop used

        callback_lists=[early_stopping,checkpoint]
    else:
        callback_lists=[checkpoint]
    
    trainX_t.shape=(trainX_t.shape[0],input_row,input_col)
    if(valX is not None):
        valX_t.shape=(valX_t.shape[0],input_row,input_col)
    
    if compiletimes==0:
         filtersize1=1
         filtersize2=9
         filtersize3=10
         filter1=200
         filter2=150
         filter3=200
         dropout1=0.75
         dropout2=0.75
         dropout4=0.75
         dropout5=0.75
         dropout6=0.25
         L1CNN=0
         l2value = 0.001
         nb_classes=2
         actfun="relu"; 
         optimization='adam';
         attentionhidden_x=10
         attentionhidden_xr=8
         attention_reg_x=0.151948
         attention_reg_xr=2
         dense_size1=149
         dense_size2=8
         dropout_dense1=0.298224
         dropout_dense2=0
         
         input2 = Input(shape=(input_row,input_col))

         x = conv.Convolution1D(filter1, filtersize1,init='he_normal',W_regularizer= l2(l2value),border_mode="same")(input2) 
         x = Dropout(dropout1)(x)
         x1 = Activation(actfun)(x)

         y1 = conv.Convolution1D(filter2,filtersize2,init='he_normal',W_regularizer= l2(l2value),border_mode="same")(x1)
         y1 = Dropout(dropout2)(y1)
         y1 = Activation(actfun)(y1)

         y2 = conv.Convolution1D(filter2,6,init='he_normal',W_regularizer= l2(l2value),border_mode="same")(x1)
         y2 = Dropout(dropout2)(y2)
         y2 = Activation(actfun)(y2)

         y3 = conv.Convolution1D(filter2,3,init='he_normal',W_regularizer= l2(l2value),border_mode="same")(x1)
         y3 = Dropout(dropout2)(y3)
         y3 = Activation(actfun)(y3)

         mergeY = merge([y1, y2, y3], mode='concat', concat_axis=-1)
         mergeY = Dropout(0.75)(mergeY)


         z1 = conv.Convolution1D(filter3,filtersize3,init='he_normal',W_regularizer= l2(l2value),border_mode="same")(mergeY)
         z1 = Activation(actfun)(z1)

         z2 = conv.Convolution1D(filter3,5,init='he_normal',W_regularizer= l2(l2value),border_mode="same")(mergeY)
         z2 = Activation(actfun)(z2)

         z3 = conv.Convolution1D(filter3,15,init='he_normal',W_regularizer= l2(l2value),border_mode="same")(mergeY)
         z3 = Activation(actfun)(z3)

         x_reshape=core.Reshape((z1._keras_shape[2],z1._keras_shape[1]))(z1)
         
         x = Dropout(dropout4)(z1)
         x_reshape=Dropout(dropout5)(x_reshape)
         
         decoder_x = Attention(hidden=attentionhidden_x,activation='linear',init='he_normal',W_regularizer=l1(attention_reg_x)) # success  
         decoded_x=decoder_x(x)
         output_x = myFlatten(x._keras_shape[2])(decoded_x)

         decoder_xr = Attention(hidden=attentionhidden_xr,activation='linear',init='he_normal',W_regularizer=l1(attention_reg_xr))
         decoded_xr=decoder_xr(x_reshape)

         output_xr = myFlatten(x_reshape._keras_shape[2])(decoded_xr)

         x_reshape=core.Reshape((z2._keras_shape[2],z2._keras_shape[1]))(z2)
         
         x = Dropout(dropout4)(z2)
         x_reshape=Dropout(dropout5)(x_reshape)
         
         decoder_x = Attention(hidden=attentionhidden_x,activation='linear',init='he_normal',W_regularizer=l1(attention_reg_x)) # success  
         decoded_x=decoder_x(x)
         output_x2 = myFlatten(x._keras_shape[2])(decoded_x)

         decoder_xr = Attention(hidden=attentionhidden_xr,activation='linear',init='he_normal',W_regularizer=l1(attention_reg_xr))
         decoded_xr=decoder_xr(x_reshape)

         output_xr2 = myFlatten(x_reshape._keras_shape[2])(decoded_xr)

         x_reshape=core.Reshape((z3._keras_shape[2],z3._keras_shape[1]))(z3)
         
         x = Dropout(dropout4)(z3)
         x_reshape=Dropout(dropout5)(x_reshape)
         
         decoder_x = Attention(hidden=attentionhidden_x,activation='linear',init='he_normal',W_regularizer=l1(attention_reg_x)) # success  
         decoded_x=decoder_x(x)
         output_x3 = myFlatten(x._keras_shape[2])(decoded_x)

         decoder_xr = Attention(hidden=attentionhidden_xr,activation='linear',init='he_normal',W_regularizer=l1(attention_reg_xr))
         decoded_xr=decoder_xr(x_reshape)

         output_xr3 = myFlatten(x_reshape._keras_shape[2])(decoded_xr)

         output=merge([output_x,output_xr,output_x2,output_xr2,output_x3,output_xr3],mode='concat')

         output=Dropout(dropout6)(output)
         output=Dense(dense_size1,init='he_normal',activation='relu')(output)
         output=Dropout(dropout_dense1)(output)
         output=Dense(dense_size2,activation="relu",init='he_normal')(output)
         output=Dropout(dropout_dense2)(output)
         out=Dense(nb_classes,init='he_normal',activation='softmax')(output)
         cnn=Model(input2,out)
         cnn.compile(loss='binary_crossentropy',optimizer=optimization,metrics=['accuracy'])
         
    else:
         cnn=compilemodels
    
    if(predict is False):
         if(background_weights is not None and compiletimes==0): #for the first time
            if not for_transfer:
                 cnn.load_weights(background_weights);
            else:
                 cnn2=Model(input2,out)
                 cnn2.compile(loss='binary_crossentropy',optimizer=optimization,metrics=['accuracy'])

                 cnn2.load_weights(background_weights);
                 for l in range((len(cnn2.layers)-n_transfer_layer)): #the last cnn is not included
                    cnn.layers[l].set_weights(cnn2.layers[l].get_weights())
                    cnn.layers[l].trainable= False  # for frozen layer
                 cnn.compile(loss='binary_crossentropy',optimizer=optimization,metrics=['accuracy'])
         
         if(valX is not None):
             if(n_earlystop is None):
               fitHistory = cnn.fit(trainX_t, trainY, batch_size=batch_size, epochs=epochs,validation_data=(valX_t, valY))
               
             else:
               fitHistory = cnn.fit(trainX_t, trainY, batch_size=batch_size, epochs=epochs, validation_data=(valX_t, valY), callbacks=callback_lists)
         else:
             fitHistory = cnn.fit(trainX_t, trainY, batch_size=batch_size, epochs=epochs)
    
    return cnn
