# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 16:24:45 2018

@author: 1papmeie
"""

from statistics import plotWrapper
from dataLoader import UrbanSound8KdataPrep


if __name__ == "__main__":
    loader = UrbanSound8KdataPrep(SR=16000)
    normalTrainData = loader.trainData
    signal_rate = loader.rate
    minTimeStretch = 0.9
    maxTimeStretch = 1.1
    stepsTimeStretch = 2
    minPitchShift = 0.9
    maxPitchShift = 1.1
    stepsPitchShift = 2
    minLevelAdjust = 1.0
    maxLevelAdjust = 4.0
    stepsLevelAdjust = 4
    #augmentedTrainData = augment(normalTrainData,signal_rate,minTimeStretch,maxTimeStretch,stepsTimeStretch,minPitchShift,maxPitchShift,stepsPitchShift,minLevelAdjust,maxLevelAdjust,stepsLevelAdjust)
    augmentedTrainData = normalTrainData
    
    lengthScene = 1
    eventsPerScene = 30
    numberMelBins = 50
    nfft=2048
    hop=1024
    trainSet = loader.prepareTrainSet(augmentedTrainData,lengthScene=lengthScene,lengthExample=1,hopExample=1,
                                      eventsPerScene=eventsPerScene,N_FFT=nfft,hop=hop,randSeed=1337,
                                      log_power=True,deriv=False,frameEnergy=False,asFeatureVector=False,
                                      n_mels=numberMelBins)
    validationSet = loader.prepareValSet(lengthScene=lengthScene,lengthExample=1,hopExample=1,eventsPerScene=eventsPerScene,
                                         N_FFT=nfft,hop=hop,randSeed=1337,log_power=True,asFeatureVector=False,
                                         deriv=False,frameEnergy=False,n_mels=numberMelBins)
    trainMeanBeforeNorm = loader.trainMean
    trainStdBeforeNorm = loader.trainStd
    trainX = trainSet[0]
    valX = validationSet[0]
    #Plot train set mean,std,min,max
    plotWrapper(trainX,'UrbanSound8K_train',log=False)
    #Plot valid
    plotWrapper(valX,'UrbanSound8K_val',log=False)