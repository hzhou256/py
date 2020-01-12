from DeepCleave_CNN import DeepCleave_CNN
from DProcess import DL_encoding
import pandas as pd
import numpy as np
import keras.models as models
from keras.models import Model

def BootStrapping(train_all_info,output_model_name,window=15,s_rate=0.9,n_BootStrapping=1,epochs=500,n_earlystop=None,coding_mode='Onehot',n_transfer_layer=1,background_weights=None,for_transfer=False):
  trainX = train_all_info
  train_pos=trainX[np.where(trainX[:,0]==1)]
  train_neg=trainX[np.where(trainX[:,0]!=1)]
  train_pos=pd.DataFrame(train_pos)
  train_neg=pd.DataFrame(train_neg)

  train_pos_s=train_pos.sample(train_pos.shape[0]) #shuffle train pos
  train_neg_s=train_neg.sample(train_neg.shape[0]) #shuffle train neg
  slength=int(train_pos.shape[0]*s_rate)
  nclass=int(train_neg.shape[0]/slength)

  a=int(train_pos.shape[0]*0.9)
  b=train_neg.shape[0]-(train_pos.shape[0] - a)
  train_pos_s=train_pos[0:a]
  train_neg_s=train_neg[0:b]
  
  val_pos=train_pos[a:]
  val_neg=train_neg[b:]

  val_all=pd.concat([val_pos,val_neg])
  
  val_all2=val_all.iloc[:,0:2*window+1]
  vids=val_all.iloc[:,2*window+1]
  vposes=val_all.iloc[:,2*window+2]
  vfocuses=val_all.iloc[:,2*window+3]
  
  valX1,valY1 = DL_encoding(val_all2.as_matrix(),coding_mode=coding_mode)
  slength=int(train_pos_s.shape[0]*s_rate) 
  nclass=int(train_neg_s.shape[0]/slength)

  for I in range(n_BootStrapping):
    train_neg_s=train_neg_s.sample(train_neg_s.shape[0]) #shuffle neg sample
    train_pos_ss=train_pos_s.sample(slength)
    for t in range(nclass):
        train_neg_ss=train_neg_s[(slength*t):(slength*t+slength)]
        train_all=pd.concat([train_pos_ss,train_neg_ss])

        train_all2=train_all.iloc[:,0:2*window+1]
        tids=train_all.iloc[:,2*window+1]
        tposes=train_all.iloc[:,2*window+2]
        tfocuses=train_all.iloc[:,2*window+3]

        trainX1,trainY1 = DL_encoding(train_all2.as_matrix(),coding_mode=coding_mode) 
        if t==0:
            models=DeepCleave_CNN(output_model_name,trainX1,trainY1,valX1,valY1,epochs=epochs,n_earlystop=n_earlystop,n_transfer_layer=n_transfer_layer,background_weights=background_weights,for_transfer=for_transfer,compiletimes=t)
        else:
            models=DeepCleave_CNN(output_model_name,trainX1,trainY1,valX1,valY1,epochs=epochs,n_earlystop=n_earlystop,n_transfer_layer=n_transfer_layer,background_weights=background_weights,for_transfer=for_transfer,compiletimes=t,compilemodels=models)
            
        print('The weights of model assigned for '+str(t)+' bootstrap.\n')
  
  return models
