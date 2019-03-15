# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 13:25:00 2018

@author: 1papmeie
"""

from statistics import plotWrapper
from dataLoader import DCASE2dataPrep
from audioProcessing import augment


if __name__ == "__main__":
    loader = DCASE2dataPrep(SR=30000)
    normalTrainData = loader.trainData
    signal_rate = loader.rate
    minTimeStretch = 0.9
    maxTimeStretch = 1.1
    stepsTimeStretch = 2
    minPitchShift = -2.
    maxPitchShift = 2.
    stepsPitchShift = 2
    minLevelAdjust = 0.0
    maxLevelAdjust = 2.0
    stepsLevelAdjust = 3
    augmentedTrainData = augment(normalTrainData,signal_rate,minTimeStretch,maxTimeStretch,stepsTimeStretch,minPitchShift,maxPitchShift,stepsPitchShift,minLevelAdjust,maxLevelAdjust,stepsLevelAdjust)
    
    lengthScene = 10
    eventsPerScene = 5
    numberMelBins = 80
    nfft=2048
    hop=512
    lenExample = 10
    hopExample = 10
    trainSet = loader.prepareTrainSet(augmentedTrainData,lengthScene=lengthScene,lengthExample=lenExample,hopExample=hopExample,
                                      eventsPerScene=eventsPerScene,N_FFT=nfft,hop=hop,randSeed=1337,
                                      log_power=True,deriv=False,frameEnergy=False,asFeatureVector=False,temporalCompression=True,
                                      n_mels=numberMelBins,predDelay=0,zNorm=True)
    validationSet = loader.prepareValSet(lengthScene=lengthScene,lengthExample=lenExample,hopExample=hopExample,eventsPerScene=eventsPerScene,
                                         N_FFT=nfft,hop=hop,randSeed=1337,log_power=True,asFeatureVector=False,temporalCompression=True,
                                         deriv=False,frameEnergy=False,n_mels=numberMelBins,
                                         predDelay=0,zNorm=True)
    trainMeanBeforeNorm = loader.trainMean
    trainStdBeforeNorm = loader.trainStd
    trainX = trainSet[0]
    valX = validationSet[0]
    #Plot train set mean,std,min,max
    plotWrapper(trainX,'DCASE2_train',log=False)
    #Plot valid
    plotWrapper(valX,'DCASE2_val',log=False)
