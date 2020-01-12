#coding=utf-8
import sys
import os
import pandas as pd
import numpy as np
import argparse

#import tensorflow as tf
#import keras.backend.tensorflow_backend as KTF
import os
import csv

from Bootstrapping import BootStrapping
from EXtract_fragment import extract_fragment_for_train, extract_fragment_for_predict
from DeepCleave_CNN import DeepCleave_CNN
from DProcess import DL_encoding

'''
# CPU

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

'''

'''
# GPU

os.environ['CUDA_VISIBLE_DEVICES']='0'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.92
session = tf.Session(config=config)

KTF.set_session(session)

'''

def base_main(args):

    input_data=args.input_data
    output_prefix=args.output_prefix
    n_models=args.n_models
    window=args.window
    epochs=args.epochs
    n_earlystop=args.n_earlystop
    background_weights=args.background_weights
    n_transfer_layer=args.n_transfer_layer
    focus_residues=args.focus_residues.split(',')
    output_model=output_prefix+str('_HDF5model')
    output_parameter=output_prefix+str('_parameters')

    with open(output_parameter, 'w') as output:
      output.write('%d\t%d\t%s\tbase' % (n_models,window,args.focus_residues))

    trainfrag,tids,tposes,tfocuses=extract_fragment_for_train(input_data,window,'-',focus=focus_residues)
    
    train_all_info = pd.concat([trainfrag,tids,tposes,tfocuses],axis=1)
    
    for bt in range(n_models):
        output_model_name=output_model+'_class'+str(bt)
        models=BootStrapping(train_all_info.as_matrix(),output_model_name,
                                                 s_rate=0.9,n_BootStrapping=1,epochs=epochs,n_earlystop=n_earlystop,
                                                 background_weights=background_weights,
                                                 n_transfer_layer=n_transfer_layer,
                                                 coding_mode='Onehot')

        #models.save_weights(output_model+'_class'+str(bt),overwrite=True)
        
def transfer_main(args):

    input_data=args.input_data;
    background_prefix=args.background_prefix;
    background_model=background_prefix+str('_HDF5model')
    background_parameter=background_prefix+str('_parameters')
    n_transfer_layer=args.n_transfer_layer
    
    with open(background_parameter,'r') as f:
      parameters=f.read()
    
    output_prefix=args.output_prefix
    n_models=args.n_models
    n_models=int(n_models)
    n_models_init=parameters.split('\t')[0]
    n_models_init=int(n_models_init)
    window=parameters.split('\t')[1]
    window=int(window)
    focus_residues_specify=args.focus_residues;

    if(focus_residues_specify is None):
       focus_residues=parameters.split('\t')[2]
    else:
       focus_residues=focus_residues_specify

    epochs=args.epochs
    n_earlystop=args.n_earlystop
    output_model=output_prefix+str('_HDF5model')
    output_parameter=output_prefix+str('_parameters')
    
    with open(output_parameter,'w') as output:
      output.write('%d\t%d\t%s\ttransfer\t%d\n' % (n_models,window,focus_residues,n_models_init))

    focus_residues=focus_residues.split(',')
    trainfrag,tids,tposes,tfocuses=extract_fragment_for_train(input_data,window,'-',focus=focus_residues)
    train_all_info = pd.concat([trainfrag,tids,tposes,tfocuses],axis=1)
    
    for ini in range(n_models_init):
        background=background_model+'_class'+str(ini)
        for bt in range(n_models):
            output_model_name=output_model+'_ini'+str(ini)+'_class'+str(bt)
            models=BootStrapping(train_all_info.as_matrix(),output_model_name,window=window,
                                                     s_rate=0.9,n_BootStrapping=1,epochs=epochs,
                                                     n_earlystop=n_earlystop,
                                                     for_transfer=True,
                                                     coding_mode='Onehot',
                                                     background_weights=background,
                                                     n_transfer_layer=n_transfer_layer)
            #models.save_weights(output_model+'_ini'+str(ini)+'_class'+str(bt),overwrite=True)

def predict_main(args):
    
    input_data=args.input_data
    output_file=args.output_file
    focus_residues=args.focus_residues.split(',')
    model_prefix=args.model_prefix

    if model_prefix is None:
        print('Please specify the prefix for an existing model by -model_prefix!\n')
        exit()
    else:
        model=model_prefix+str('_HDF5model')
        parameter=model_prefix+str('_parameters')

        with open(parameter,'r') as fp:
            parameters=fp.read()

        n_class=int(parameters.split('\t')[0])
        window=int(parameters.split('\t')[1])
        focus_residues=parameters.split('\t')[2]
        focus_residues=focus_residues.split(',')

        test_frag,test_ids,test_poses,test_focuses=extract_fragment_for_predict(input_data,window,'-',focus=focus_residues)
        
        testAllInfo = pd.concat([test_frag,test_ids,test_poses,test_focuses],axis=1)

        testAll = testAllInfo.as_matrix()
        test_pos=testAll[np.where(testAll[:,0]==1)]
        test_neg=testAll[np.where(testAll[:,0]!=1)]
        test_pos=pd.DataFrame(test_pos)
        test_neg=pd.DataFrame(test_neg)
        test_pos_length=int(test_pos.shape[0])

        times = 0
        while(times < 5):

          test_neg_shuffle = test_neg.sample(test_neg.shape[0])
          test_neg_samelen = test_neg_shuffle[0:test_pos_length]

          test_all=pd.concat([test_pos,test_neg_samelen])
          test_all2=test_all.iloc[:,0:2*window+1]
          t_ids=test_all.iloc[:,2*window+1]
          t_poses=test_all.iloc[:,2*window+2]
          t_focuses=test_all.iloc[:,2*window+3]

          testX,testY = DL_encoding(test_all2.as_matrix(),coding_mode='Onehot')
          predict_proba=np.zeros((testX.shape[0],2))
          
          models=DeepCleave_CNN('default',testX,testY,epochs=1,predict=True)# only to get config
          
          if(parameters.split('\t')[3]=='transfer'):
               n_class_ini=int(parameters.split('\t')[4])
               for ini in range(n_class_ini):
                   for bt in range(n_class):
                        models.load_weights(model+'_ini'+str(ini)+'_class'+str(bt))
                        predict_proba+=models.predict(testX)
               
          else:
             n_class_ini=1;
             for bt in range(n_class):
                 models.load_weights(model+'_class'+str(bt))
                 predict_proba+=models.predict(testX)
          
          predict_proba=predict_proba/(n_class*n_class_ini);
          t_poses=t_poses+1;
          results=np.column_stack((testY[:,1],predict_proba[:,1]))
          result=pd.DataFrame(results)
          result.to_csv(output_file+'_'+ str(times) +'_predict.txt', index=False, header=None, sep='\t',quoting=csv.QUOTE_NONNUMERIC)
          print('Successfully predicted from DeepCleave models !\n')

          times += 1

         
if __name__ == '__main__':

    parser=argparse.ArgumentParser(description='DeepCleave: a deep learning predictor for caspase and matrix metalloprotease substrates and cleavage sites')
    
    required = parser.add_argument_group('Required arguments')

    required.add_argument('-input_data', type=str, help='Training data in fasta format, where the positive sites are followed by "#".', required=True)
    required.add_argument('-focus_residues', type=str, help='DeepCleave will train for different Residue types, with multiple Residues separated by commas.', required=True)
    required.add_argument('-DeepCleave_type', required=True,choices=['base', 'transfer', 'predict'], help='Select the type of DeepCleave to run')

    DeepCleaveOptional = parser.add_argument_group('DeepCleave optional arguments')

    DeepCleaveOptional.add_argument('-n_models', type=int, help='Number of models to be trained.', required=False, default=5)
    DeepCleaveOptional.add_argument('-window', type=int, help='Window size', required=False, default=15)
    DeepCleaveOptional.add_argument('-epochs', type=int, help='Number of epoches for one bootstrap step.', required=False, default=500)
    DeepCleaveOptional.add_argument('-n_earlystop', type=int, help='When the model does not have any improvement in the n_earlystop round training, one of the bootstrap steps will end prematurely.', required=False, default=20)
    DeepCleaveOptional.add_argument('-background_weights', type=int, help='Initial weights saved in a HDF5 file.', required=False, default=None)
    DeepCleaveOptional.add_argument('-n_transfer_layer', type=int, help='The weight of the last n_transfer_layer layer of the model is randomly initialized.', required=False, default=1)
    
    baseTransferOptional = parser.add_argument_group('base and transfer arguments')
    baseTransferOptional.add_argument('-output_prefix', type=str, help='The prefix of the output files.', required=False)
      
    transferOptional = parser.add_argument_group('transfer arguments')
    transferOptional.add_argument('-background_prefix', type=str, help='The prefix of the pre-trained model.', required=False)
      
    predictOptional = parser.add_argument_group('predict arguments')
    predictOptional.add_argument('-output_file', type=str, help='The prefix of the prediction results.', required=False)
    predictOptional.add_argument('-model_prefix', type=str, help='The prefix of the trained model.', required=False,default=None)
         
    args = parser.parse_args()

    DeepCleave_type = args.DeepCleave_type

    if(DeepCleave_type == 'base'):
      base_main(args)
    elif(DeepCleave_type == 'transfer'):
      transfer_main(args)
    elif(DeepCleave_type == 'predict'):
      predict_main(args)
