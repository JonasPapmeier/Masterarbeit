# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 12:52:25 2017

@author: brummli
"""

import os

#Targets for model and logging files/folders
modelsFolder = 'models/'
logFolder = 'history/'
#Designed for use on multiple cluster blades, change to something valid for home use
#globalLog = os.path.expanduser('~')+'/Masterarbeit/programming/DCASE/experimentLog.txt'
#globalCsv = os.path.expanduser('~')+'/Masterarbeit/programming/DCASE/experimentLog.csv'
globalLog = 'experimentLog.txt '
globalCsv = 'experimentLog.csv'


DCASE2TrainFolder = 'dcase2016_task2_train_dev/dcase2016_task2_train/'
DCASE2DevFolder = 'dcase2016_task2_train_dev/dcase2016_task2_dev/sound/'
DCASE2TestFolder = 'dcase2016_task2_test_public/sound/'

UrbanSound8KBaseFolder = 'UrbanSound8K/audio'


#Default values for parser/functions
epochs=10
batchSize=16
beta=0.5

#Neural net specific
layers=3
hiddenNodes=216
lr=0.01
sigma=0.1
predictionDelay=0

#Feature extraction specific
signalRate=30000
signalRateUS8K = 16000
lengthScene=10.0
lengthExample=11
hopExample=1
numEvents=5
method='mel'
nFFT=2048
hop=512
nBin=50
augmentation=-1
logPower="store_false" #equals default=True
deriv="store_false"
frameEnergy="store_false"
max_dB=120
trainHoldoutSplit=0.1

#GMM specific
numComponents=2
GMM_tolerance=1e-3
regularization=1e-6

#KMeans specific
numCluster=11
patience=5000
reassignment=0.01
KMeans_tolerance = 0.0
KMeans_batch_size = 2000

#GWR specific
maxNodes = 5000
maxNeighbours = 10000
maxAge = 3
habThres = 0.3
insThres = 0.5
epsilonB = 0.5
epsilonN = 0.001
tauB = 0.3
tauN = 0.1

#Autoencoder extraction specific
extractionLayer=6


#Probably easier and prettier to solve via a dict or something similar
def getAugmentConfig(config=0):
    if config == 0:
        minTime = 0.9
        maxTime = 1.1
        stepsTime = 2
        minPitch = -1
        maxPitch = 1
        stepsPitch = 2
        minLevel = 0.0
        maxLevel = 2.0
        stepsLevel = 3
    elif config == 1:
        minTime = 0.9
        maxTime = 1.1
        stepsTime = 4
        minPitch = -2
        maxPitch = 2
        stepsPitch = 5
        minLevel = 1.0
        maxLevel = 2
        stepsLevel = 5
    elif config == 2:
        minTime = 1
        maxTime = 1
        stepsTime = 0
        minPitch = -2
        maxPitch = 2
        stepsPitch = 5
        minLevel = 1
        maxLevel = 2
        stepsLevel = 2
    elif config == 3:
        minTime = 0.9
        maxTime = 1.1
        stepsTime = 4
        minPitch = -2
        maxPitch = 2
        stepsPitch = 5
        minLevel = 1
        maxLevel = 5
        stepsLevel = 9
    elif config ==4:
        minTime = 1
        maxTime = 1
        stepsTime = 0
        minPitch = -2
        maxPitch = 2
        stepsPitch = 0
        minLevel = -1
        maxLevel = 1
        stepsLevel = 3
    elif config == 5:
        minTime = 1
        maxTime = 1
        stepsTime = 0
        minPitch = 0
        maxPitch = 0
        stepsPitch = 0
        minLevel = 0
        maxLevel = 2
        stepsLevel = 3
    else:
        minTime=1
        maxTime=1
        stepsTime=0
        minPitch=1
        maxPitch=1
        stepsPitch=0
        minLevel = 0
        maxLevel = 0
        stepsLevel = 1
    return (minTime, maxTime, stepsTime, minPitch, maxPitch, stepsPitch, minLevel, maxLevel, stepsLevel)
