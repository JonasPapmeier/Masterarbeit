# -*- coding: utf-8 -*-
"""
Created on Thu May 10 14:53:49 2018

@author: 1papmeie
"""

import numpy as np
import dataLoader
import audioProcessing
import GWR

if __name__ == '__main__':
    loader = dataLoader.DCASE2dataPrep(SR=16000,dev=False,test=False)
    lengthExample = 1
    hopExample = 1
    asFeatureVector= True
    #TODO: augment data
    data = loader.trainData
    minTime = 0.9
    maxTime = 1.1
    stepsTime = 0
    minPitch = 0.9
    maxPitch = 1.1
    stepsPitch = 0
    minLevel = 0.0
    maxLevel = 2.0
    stepsLevel = 3
    augmentedData = audioProcessing.augment(data,loader.rate,minTime,maxTime,stepsTime,minPitch,maxPitch,stepsPitch,minLevel,maxLevel,stepsLevel)
    loader.estimateStatistics(augmentedData, 20, deriv=False,frameEnergy=False,n_mels=25)
    datasplits = loader.splitDataByLabel(augmentedData,loader.labels)
    genExample = loader.genTrainSetExample(datasplits[0],lengthExample=lengthExample,hopExample=hopExample,
                                           asFeatureVector=asFeatureVector,deriv=False,frameEnergy=False,
                                           n_mels=25,N_FFT=2048,hop=2048,zNorm=False,minMaxNorm=True,l2Norm=False)
    examplesPerEpoch = 1000
    initial_x, initial_y, _ = next(genExample)
    exampleArray = np.zeros((examplesPerEpoch,initial_x.shape[-1]))
    targetArray = np.zeros((examplesPerEpoch,initial_y.shape[-1]))
    exampleArray[0] = initial_x
    targetArray[0] = initial_y
    for i in range(1,examplesPerEpoch):
        exampleArray[i],targetArray[i],_ = next(genExample)
    
    #Worked quite well for 40000 example per epoch
    #gwr = GWR.GWR(exampleArray,max_epochs=1,hab_threshold=0.6,insertion_threshold=0.8,
    #              epsilon_b=0.5,epsilon_n=0.1,tau_b=0.3,tau_n=0.1,MAX_NODES=3500,MAX_NEIGHBOURS=1000,
    #              MAX_AGE=3)
    #
    gwr = GWR.GWR(exampleArray,max_epochs=1,hab_threshold=0.6,insertion_threshold=0.9,
                  epsilon_b=0.25,epsilon_n=0.000001,tau_b=0.01,tau_n=0.001,MAX_NODES=10000,MAX_NEIGHBOURS=1000,
                  MAX_AGE=20)
    epochs=100//len(datasplits)
    for datasplit in range(len(datasplits)):
        genExample = loader.genTrainSetExample(datasplits[datasplit],lengthExample=lengthExample,hopExample=hopExample,
                                           asFeatureVector=asFeatureVector,deriv=False,frameEnergy=False,
                                           n_mels=25,randSeed=1337+datasplit*100,N_FFT=2048,hop=2048,
                                           zNorm=False,minMaxNorm=True,l2Norm=False)
        for epoch in range(epochs):
            for example in range(examplesPerEpoch):
                exampleArray[example],targetArray[example],_ = next(genExample)
            print('Datasplit {}: Train epoch {}'.format(datasplit,epoch))
            gwr.train(data=exampleArray,labels=targetArray)
            gwr.plotWithLabel('plots/gwr/smallTestdatasplit'+str(datasplit)+'epoch'+str(epoch)+'.png',dimReductionMethod='PCA')
            #print(len(gwr.connectedComponents))
    gwr.save('TestGWR')