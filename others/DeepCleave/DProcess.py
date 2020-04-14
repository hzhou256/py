import string
import re
import pandas as pd
import numpy as np
import keras.utils.np_utils as kutils

def Onehot_Encoding(sampleSeq3DArr):

    AADict = {}
    AADict['A'] = 0
    AADict['C'] = 1
    AADict['D'] = 2
    AADict['E'] = 3
    AADict['F'] = 4
    AADict['G'] = 5
    AADict['H'] = 6
    AADict['I'] = 7
    AADict['K'] = 8
    AADict['L'] = 9
    AADict['M'] = 10
    AADict['N'] = 11
    AADict['P'] = 12
    AADict['Q'] = 13
    AADict['R'] = 14
    AADict['S'] = 15
    AADict['T'] = 16
    AADict['V'] = 17
    AADict['W'] = 18
    AADict['Y'] = 19
    AADict['-'] =20
    AACategoryLen = len(AADict)
    
    probMatr = np.zeros((len(sampleSeq3DArr), len(sampleSeq3DArr[0]), AACategoryLen))
    
    
    sampleNo = 0
    for sequence in sampleSeq3DArr:
        AANo	 = 0
        for AA in sequence:
            
            if not AA in AADict:
                probMatr[sampleNo][AANo][AADict['-']] = 1
            else:
                index = AADict[AA]
                probMatr[sampleNo][AANo][index] = 1
            AANo += 1
        sampleNo += 1
    
    return probMatr

def DL_encoding(rawDataFrame,coding_mode='Onehot'):
    
    targetList = rawDataFrame[:, 0]
    targetArr = kutils.to_categorical(targetList)

    if(coding_mode=='Onehot'):
        sampleSeq3DArr = rawDataFrame[:, 1:]
        probMatr = Onehot_Encoding(sampleSeq3DArr)

    return probMatr, targetArr
