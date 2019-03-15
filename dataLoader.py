# -*- coding: utf-8 -*-
"""
Created on Wed May 17 16:52:34 2017

@author: 1papmeie
"""

import numpy as np
import librosa
import re
import random
import math
import evaluation
from audioProcessing import loadFolderAudio, melSpecExtraction, augment
from abc import ABC, abstractmethod
import config_
import utils
import scipy
import csv
from multiprocessing import Pool
from functools import partial
from collections import deque
import time

"""
Train data label is implicitly given by name. As we want to train sequence to sequence
every frame will be annotated with the corresponding label.

Development data has extra annotation file with labels and onset/offset times.
The onset/offsets need to be translated to framewise labels
(multiple functions?)

In long term there will be a need for data augmentation and thus maybe a generator
also possible to apply fixed data augmentation (frequency shift/time scale)
only noise should be added dynamically
"""

#TODO: fetch scripts to download datasets if they are not present 
#in the expected folder structure
            
#Responsible to load DCASE 2016 Challenge 2 training data with labels
def loadTrainDataDCASE2(trainFolder,sampling_rate):
    print('Loading train data')
    trainData = loadFolderAudio(trainFolder,SR=sampling_rate)
    #Match all strings to classes
    train = []
    #labels = []
    labels = ['cough','clearthroat','pageturn','doorslam','speech','keys','drawer','phone','keyboard','laughter','knock']
    for example in trainData:
        name = example[0]
        #Names are structurally labelxxx.wav where x is a number
        labelString = re.split('\d',name)[0]
        if labelString == 'keysDrop':   #erronuous naming between train data and annotation in dev data
            labelString = 'keys'
        if labelString in labels:
            labelNum = labels.index(labelString)
        else:
            labels.append(labelString)
            labelNum = len(labels)-1
        normExample = example[1] / np.max(np.abs(example[1]))
        train.append((labelNum,normExample))
    print('Value, Label mappings are:')
    for x in enumerate(labels):
        print(x)
    print('Finished loading train data')
    return train, labels

def loadDataUrbanSound8K(folder,sampling_rate,fold):
    """
    Loads data from UrbanSound8k dataset in single folds
    """
    
    foldFolder = folder+'/fold'+str(fold)+'/'
    print('Loading Urban Sound Data from folder', foldFolder)
    data = loadFolderAudio(foldFolder,SR=sampling_rate)
    outData = []
    labels = ['air_conditioner','car_horn','children_playing','dog_bark','drilling','engine_idling','gun_shot','jackhammer','siren','street_music']
    #Sorting of the csv would be nice, but a) we need to read the whole file first
    #b) we dont have the explicit split to sort, c) might not even be faster
    with open(folder+'/../metadata/UrbanSound8K.csv', newline='') as metaDataFile:
        labelReader = csv.DictReader(metaDataFile, delimiter=',')
        for row in labelReader:
            if int(row['fold']) == fold:
                for ind, example in enumerate(data):
                    name = example[0].replace('.mp3','.wav')
                    rawData = example[1]
                    if name == row['slice_file_name']:
                        label = int(row['classID'])
                        outData.append((label,rawData))
                        data.pop(ind)
                        break
        assert len(data)==0, 'Had some data left after searching through csv'
    return outData, labels
                    
    

def parseStringListToEventList(stringList,labels):
    eventList = []
    for s in stringList:
        parts = s.split('\t')
        onset = float(parts[0])
        offset= float(parts[1])
        label = labels.index(parts[2][0:-1])
        eventList.append((onset,offset,label))
    return eventList
    
def loadDevTestData(devTestFolder,sampling_rate=30000):
    """Loads development (i.e. validation) and test data for DCASE2 dataset"""
    print('Loading dev/test data')
    folder = devTestFolder
    devTestData = loadFolderAudio(folder,SR=sampling_rate)
    #File name is the same as in annotations folder with .wav replaced by .txt
    #annotations are in string form: onset   offset  label \n
    #needs to be later transfered to frame wise annotation
    devTest = []
    for example in devTestData:
        name = example[0]
        print(folder+'../annotation/'+name[0:len(name)-3]+'txt')
        annotationFile = open(folder+'../annotation/'+name[0:len(name)-3]+'txt',mode='r')
        annotations = []
        for line in annotationFile:
            annotations.append(line)
        devTest.append((annotations,example[1]))
    print('Finished loading dev/test data')
    return devTest

    
"""
Pre comment turn into doc later
Abstract class for data preperation
train data (list): List of pairs (int,numpy.array) containing numeric label and raw audio data as numpy array
labels (list): List of pairs (int,string) containing numeric and corresponding literal label
Assumes train data may be changed or augmented but Dev Set and Test Set are provided fully prepared only feature extraction has to be done
Params:
    SR (int): signal rate with which the audio data should be loaded
"""
class dataPrep(ABC):
    
    #TODO: init method needed for abstract superclass?
    @abstractmethod
    def __init__(self, SR=16000):
        pass
        
    """
    Prepares dataset as two numpy arrays
    assumes data to be a list of pairs (label,audio)
    Params:
        length (int): length in seconds of an example
        eventsPerExample (int): number of events put into an example
        N_FFT (int): number bins used for fast fourier transformation
        hop (int): number of frames the fourier transformation window is shifted per application
        randSeed (int): initialization number for random seed
    
    Returns:
        pair: containing numpy arrays for example x and corresponding target y
    """
    @abstractmethod
    def prepareTrainSet(self,data,lenght=120,eventsPerSample=20,N_FTT=512,hop=256,randSeed=1337):
        pass
    
    @abstractmethod
    def prepareValSet(self,N_FFT=512,hop=256):
        pass
    
    @abstractmethod
    def prepareTestSet(self,N_FFT=512,hop=256):
        pass
    
    def splitDataByLabel(self,data,labels):
        listOfLabeledLists = [[] for i in range(len(labels))]
        for example in data:
            listOfLabeledLists[example[0]].append(example)
        return listOfLabeledLists
        
#Outside of class, because we want to pass to multiprocessing pool
#inside class  would produce problems due to necessary pickle step
#TODO: Function is too large, should be refactored for single responsibility
def dcase2ProcessData(data,seed,signalRate,lengthScene,numLabel,N_FFT,hop,log_power,deriv,frameEnergy,n_mels):
    numpyRandom = np.random.RandomState(seed)
    totalLength = int(lengthScene*signalRate)
    audioLevel = 1./2.**7
    y = np.zeros((totalLength,numLabel))
    x = np.zeros(totalLength)
    actualData = np.zeros(totalLength)
    for example in data:
        if totalLength-example[1].shape[0] < 0:
            print('Warning example longer than scene, example not included')
        else:
            #Potentially overlapping
            startPoint = numpyRandom.randint(totalLength-example[1].shape[0])
            endPoint = startPoint + example[1].shape[0]
            if numpyRandom.rand() < 0.666:
                #Block overlap
                tries = 0
                maxTries = 40
                while np.any(x[startPoint:endPoint]) and tries<maxTries:
                    #Reroll startpoint
                    startPoint = numpyRandom.randint(totalLength-example[1].shape[0])
                    endPoint = startPoint + example[1].shape[0]
                    tries += 1
                if np.any(x[startPoint:endPoint]):
                    pass
                    #print('Found no non-overlapping position in {} tries'.format(maxTries))
            x[startPoint:endPoint] = x[startPoint:endPoint] + example[1][:]
            #Threshold labels so that only parts above 10% root-mean-square level are labeled
            powerLevel = np.pad(example[1]**2,2048//2,mode='constant',constant_values=0)
            powerLevel = scipy.ndimage.convolve1d(powerLevel,np.ones(2048)/2048)
            y_example = np.where(powerLevel>0.25*np.mean(powerLevel),1,0)
            #Put back to binary values
            y_example = np.where(y_example>0,1,0)
            y_example = y_example[2048//2:-2048//2]
            y[startPoint:endPoint,example[0]] = 1#y_example
            actualData[startPoint:endPoint] = 1
    actualData = np.nonzero(actualData)
    #x = x[actualData]
    #y = y[actualData]
    noiseLevel = 1.0
    #noiseLevel = 0.0625
    #The resampling procedure unfortunately changes the frequency distribution
    #it is therefore necessary to create the noise at the original dataset rate and resample it as well
    preResampleShape = int(np.ceil(x.shape[0] / (signalRate*1.0))*44100)
    preResampleShape = preResampleShape + preResampleShape%2
    whiteNoise = numpyRandom.normal(scale=1,size=preResampleShape)
    whiteNoise = whiteNoise/np.max(np.abs(whiteNoise))*(noiseLevel/3.)
    #Brown noise, implementation taken from:
    #https://github.com/python-acoustics/python-acoustics/blob/master/acoustics/generator.py, Last accessed: 16.04.2018, 18:09
    uneven = preResampleShape%2
    randSpec = numpyRandom.randn(preResampleShape//2+1+uneven) + 1j * numpyRandom.randn(preResampleShape//2+1+uneven)
    specFilter = (np.arange(len(randSpec))+1)# Filter
    #Cut off below a given Hz
    cutOff = 5 + numpyRandom.rand()*40
    specFilter[0:int(len(randSpec)/44100*cutOff)] = len(randSpec)+1
    brownNoise = (np.fft.irfft(randSpec/specFilter)).real
    brownNoise = brownNoise/np.max(np.abs(brownNoise))*noiseLevel
    combinedNoise = whiteNoise + brownNoise
    resampledNoise = librosa.core.resample(combinedNoise,44100,signalRate)
    x[:] = x[:] + resampledNoise[:x.shape[0]] #When we cut out pure silence parts we need to have the same shape here
    x = x * audioLevel
    melX = melSpecExtraction(x,sr=signalRate,N_FFT=N_FFT,HOP=hop,center=True,log_power=log_power,deriv=deriv,frameEnergy=frameEnergy,n_mels=n_mels)
    y = y[::hop,:]
    if melX.shape[0] != y.shape[0]:
        y = np.append(y,np.zeros((melX.shape[0]-y.shape[0],y.shape[-1])),axis=0)
    y_ae = melX
    assert melX.shape[0] == y.shape[0]
    return (melX,y,y_ae)

def sceneToExamples(scene,lengthExample,hopExample, trainMean, trainStd, trainMeanTarget, trainStdTarget):
    sceneX, sceneY, sceneY_AE = scene
    sceneX = (sceneX - trainMean) / trainStd
    sceneY_AE = (sceneY_AE - trainMeanTarget) / trainStdTarget
    examplesX = utils.sliding_window(sceneX, lengthExample, stepsize=hopExample,axis=0)
    examplesX = np.swapaxes(examplesX, 1, 2)
    examplesY = utils.sliding_window(sceneY, lengthExample, stepsize=hopExample,axis=0)
    examplesY = np.swapaxes(examplesY, 1, 2)
    examplesY_AE = utils.sliding_window(sceneY_AE, lengthExample, stepsize=hopExample,axis=0)
    examplesY_AE = np.swapaxes(examplesY_AE, 1, 2)
    examplesY = np.sum(examplesY,axis=1)
    examplesY = np.where(examplesY>lengthExample//2,1,0)
    
    randGen = np.random.RandomState(1337)
    randIndex = np.arange(examplesX.shape[0])
    randGen.shuffle(randIndex)
    examplesX = examplesX[randIndex]
    examplesY = examplesY[randIndex]
    examplesY_AE = examplesY_AE[randIndex]
    
    examplesX = np.split(examplesX,examplesX.shape[0],axis=0)
    examplesY = np.split(examplesY,examplesY.shape[0],axis=0)
    examplesY_AE = np.split(examplesY_AE,examplesY_AE.shape[0],axis=0)
    return (examplesX,examplesY,examplesY_AE)

def getSceneExamples(data,seed,signalRate,lengthScene,numLabel,N_FFT,hop,log_power,deriv,frameEnergy,n_mels,lengthExample,hopExample,trainMean,trainStd,trainMeanTarget,trainStdTarget):
    scene = dcase2ProcessData(data,seed,signalRate,lengthScene,numLabel,N_FFT,hop,log_power,deriv,frameEnergy,n_mels)
    examples = sceneToExamples(scene,lengthExample,hopExample,trainMean,trainStd,trainMeanTarget,trainStdTarget)
    return examples
    
#TODO: multiple refactors necessary: methods too long, contains methods for full processing and scene/batchwise generation
#some parts of code are repeated    
class DCASE2dataPrep(dataPrep):
    trainFolder = config_.DCASE2TrainFolder
    devFolder = config_.DCASE2DevFolder
    testFolder = config_.DCASE2TestFolder
    
    def __init__(self, SR=16000,dev=True,test=True,stats=None):
        self.rate = SR
        self.trainData, self.labels = loadTrainDataDCASE2(DCASE2dataPrep.trainFolder,sampling_rate=SR)
        self.holdoutData = []
        
        #Creates holdout split by taking x samples per label
        for label in range(len(self.labels)):
            takenLabelExamples = 0
            maxLabelExamples = len(self.trainData)/len(self.labels) * config_.trainHoldoutSplit
            usedExamplesIndex = []
            for num, example in enumerate(self.trainData):
                if example[0] == label:
                    usedExamplesIndex.append(num)
                    takenLabelExamples += 1
                if takenLabelExamples >= maxLabelExamples:
                    break
            for num in reversed(usedExamplesIndex):
                self.holdoutData.append(self.trainData.pop(num))
        self.devData=None
        if dev:
            self.devData = loadDevTestData(DCASE2dataPrep.devFolder,sampling_rate=SR)
        self.testData=None
        if test:
            self.testData = loadDevTestData(DCASE2dataPrep.testFolder,sampling_rate=SR)
        if stats:
            self.trainMean, self.trainStd, self.trainMin, self.trainMax, self.trainMeanTarget, self.trainStdTarget = stats
        else:
            self.trainMean = None
            self.trainStd = None
            self.trainMin = None
            self.trainMax = None
            self.trainMeanTarget = None
            self.trainStdTarget = None

    def setTrainData(self,data):
        self.trainData = data
        
    def setLabelList(self, labelList=[(0,'clearthroat'),(1,'cough'),(2,'doorslam'),(3,'drawer'),(4,'keyboard'),(5,'keysDrop'),(6,'knock'),(7,'laughter'),(8,'pageturn'),(9,'phone'),(10,'speech')]):
        self.labels = labelList
        
    def prepareTrainSet(self,data,lengthScene=config_.lengthScene,lengthExample=config_.lengthExample,
                        hopExample=config_.hopExample,asFeatureVector=False,temporalCompression=False,eventsPerScene=config_.numEvents,
                        N_FFT=config_.nFFT,hop=config_.hop,randSeed=1337,log_power=True,deriv=True,
                        frameEnergy=True,n_mels=config_.nBin,max_dB=120,predDelay=config_.predictionDelay,
                        shuffle=True,zNorm=True,minMaxNorm=False,l2Norm=False):
        random.seed(a=randSeed) #Initialize Random to be reproduceable
        data = data.copy() #Copy list as shuffle is inplace
        random.shuffle(data)
        
        totalScenes = math.ceil(len(data)/eventsPerScene)
        data.extend(data[0:(eventsPerScene-len(data)%eventsPerScene)%eventsPerScene])
        assert len(data)/eventsPerScene == totalScenes
        splitPerScene = [(data[i:i+eventsPerScene],randSeed+i) for i in np.arange(0,len(data),eventsPerScene)]
        
        featureSpecificProcessFunc = partial(dcase2ProcessData,numLabel=len(self.labels),signalRate=self.rate,lengthScene=lengthScene,
                                             N_FFT=N_FFT,hop=hop,log_power=log_power,deriv=deriv,
                                             frameEnergy=frameEnergy,n_mels=n_mels)
        with Pool() as pool:
            processedData = pool.starmap(featureSpecificProcessFunc,splitPerScene)
        xList = []
        yList = []
        y_aeList = []
        for scene in processedData:
            xList.append(np.reshape(scene[0],(-1,scene[0].shape[-1])))
            yList.append(np.reshape(scene[1],(-1,scene[1].shape[-1])))
            y_aeList.append(np.reshape(scene[2],(-1,scene[2].shape[-1])))
        x = np.concatenate(xList,axis=0)
        y = np.concatenate(yList,axis=0)
        y_ae = np.concatenate(y_aeList,axis=0)
        #TODO: look into resize instead of reshape to automatically fill with repeats of data or zeros
        flat_x = np.reshape(x,(-1,x.shape[-1]))
        flat_y = np.reshape(y,(-1,y.shape[-1]))
        flat_y_ae = np.reshape(y_ae,(-1,y_ae.shape[-1]))
        if lengthExample-(flat_x.shape[0]%lengthExample) > lengthExample/2:
            #If the last example would be filled by more than half with real data
            #Append zeros to flat_x and flat_y, so that lengthExample is a divisor of length of flat_x/flat_y
            flat_x = np.append(flat_x,flat_x[:lengthExample-(flat_x.shape[0]%lengthExample),:],axis=0)
            flat_y = np.append(flat_y,flat_y[:lengthExample-(flat_y.shape[0]%lengthExample),:],axis=0)
            flat_y_ae = np.append(flat_y_ae,flat_y_ae[:lengthExample-(flat_y_ae.shape[0]%lengthExample),:],axis=0)
        else:
            #Otherwise cut data short
            flat_x = flat_x[0:lengthExample*(flat_x.shape[0]//lengthExample),:]
            flat_y = flat_y[0:lengthExample*(flat_y.shape[0]//lengthExample),:]
            flat_y_ae = flat_y_ae[0:lengthExample*(flat_y_ae.shape[0]//lengthExample),:]
        #TODO: if resize is used we can use np.mean over multiple axis via a tuple
        if zNorm or minMaxNorm:
            if self.trainMean is None:
                self.trainMean = np.mean(flat_x,axis=0,keepdims=True)
                self.trainMeanTarget = np.mean(flat_y_ae,axis=0,keepdims=True)
            if self.trainStd is None:
                trainStd = np.std(flat_x,axis=0,keepdims=True)+1e-12
                #trainStd = np.where(trainStd==0,1e-12,trainStd)
                trainStdTarget = np.std(flat_y_ae,axis=0,keepdims=True)+1e-12
                self.trainStd = trainStd
                self.trainStdTarget = trainStdTarget
            if self.trainMax is None:
                self.trainMax = np.max(flat_x,axis=0,keepdims=True)
            if self.trainMin is None:
                self.trainMin = np.min(flat_x,axis=0,keepdims=True)
        
        if zNorm:
            flat_x = flat_x - self.trainMean
            flat_x = flat_x / self.trainStd 
            flat_y_ae = flat_y_ae - self.trainMeanTarget
            flat_y_ae = flat_y_ae / self.trainStdTarget
        if minMaxNorm:
            flat_x = ((flat_x-self.trainMin)/(self.trainMax-self.trainMin))*2-1
            flat_y_ae = ((flat_y_ae-self.trainMin)/(self.trainMax-self.trainMin))*2-1

        if (asFeatureVector or temporalCompression) and lengthExample>1:
            paddingArray = np.zeros(((int(np.floor(lengthExample/2))),flat_x.shape[-1]))
            paddingArray2 = np.zeros((int(np.ceil((lengthExample/2)))-1,flat_x.shape[-1]))
            flat_x = np.concatenate((paddingArray,flat_x,paddingArray2),axis=0)
            paddingArray = np.zeros(((int(np.floor(lengthExample/2))),flat_y.shape[-1]))
            paddingArray2 = np.zeros((int(np.ceil((lengthExample/2)))-1,flat_y.shape[-1]))
            flat_y = np.concatenate((paddingArray,flat_y,paddingArray2),axis=0)
            paddingArray = np.zeros(((int(np.floor(lengthExample/2))),flat_y_ae.shape[-1]))
            paddingArray2 = np.zeros((int(np.ceil((lengthExample/2)))-1,flat_y_ae.shape[-1]))
            flat_y_ae = np.concatenate((paddingArray,flat_y_ae,paddingArray2),axis=0)
        x = utils.sliding_window(flat_x, lengthExample, stepsize=hopExample,axis=0)
        x = np.swapaxes(x, 1, 2)
        y = utils.sliding_window(flat_y, lengthExample, stepsize=hopExample,axis=0)
        y = np.swapaxes(y, 1, 2)
        y_ae = utils.sliding_window(flat_y_ae, lengthExample, stepsize=hopExample,axis=0)
        y_ae = np.swapaxes(y_ae, 1, 2)
        if asFeatureVector:
            x = np.reshape(x,(x.shape[0],-1))
            y_ae = np.reshape(y_ae,(y_ae.shape[0],-1))
            y = np.sum(y,axis=1)
            y = np.where(y>lengthExample//2,1,0)
        if temporalCompression:
            y = np.sum(y,axis=1)
            y = np.where(y>lengthExample//2,1,0)
        #has to be done after stacking frames
        if l2Norm:
            x = x / np.sqrt(np.sum(x**2,axis=-1,keepdims=True))
            y_ae = y_ae / np.sqrt(np.sum(y_ae**2,axis=-1,keepdims=True))
        if shuffle:
            randGen = np.random.RandomState(1337)
            randIndex = np.arange(x.shape[0])
            randGen.shuffle(randIndex)
            x = x[randIndex]
            y = y[randIndex]
            y_ae = y_ae[randIndex]
        return (x,y,y_ae)
        
    def genScene(self,data,lengthScene=config_.lengthScene,randGen=np.random.RandomState(1337),eventsPerScene=config_.numEvents,N_FFT=config_.nFFT,hop=config_.hop,log_power=True,deriv=True,frameEnergy=True,n_mels=config_.nBin,autoencoder=False):
        totalLength = int(lengthScene*self.rate)
        x = np.zeros(totalLength)
        y_ae = np.zeros(totalLength)
        y = np.zeros((totalLength,len(self.labels)))
        audioLevel = 1./2.**7
        if randGen.random_sample() < 0.333:
            allowOverlap=False
        else:
            allowOverlap=True
        for i in range(eventsPerScene):
            example = data[randGen.choice(len(data))]
            if totalLength-example[1].shape[0] < 0:
                print('Warning example longer than scene, example ignored')
            else:
                startPoint = randGen.choice(totalLength-example[1].shape[0])
                endPoint = startPoint+example[1].shape[0]
                if not allowOverlap:
                    tries = 0
                    while np.any(x[startPoint:endPoint]) and tries<20:
                        #Reroll startpoint
                        startPoint = randGen.choice(totalLength-example[1].shape[0])
                        endPoint = startPoint+example[1].shape[0]
                        tries += 1
                    if np.any(x[startPoint:endPoint]):
                        print('Found no non-overlapping position in {} tries'.format(tries))
                x[startPoint:endPoint] = x[startPoint:endPoint] + example[1][:]
                #Threshold labels so that only parts above 10% root-mean-square level are labeled
                y_example = np.where(np.sqrt(example[1]**2)>0.1*np.sqrt(np.mean(example[1]**2)),1,0)
                #Smooth the result
                y_example = scipy.ndimage.convolve1d(y_example,np.ones(150))
                #Put back to 0-1 values
                y_example = np.where(y_example>0,1,0)
                y[startPoint:endPoint,example[0]] = y_example
        #Always compute denoising autoencoder output target
        y_ae = x * audioLevel
        y_ae = melSpecExtraction(y_ae,sr=self.rate,N_FFT=N_FFT,HOP=hop,center=True,log_power=log_power,deriv=deriv,frameEnergy=frameEnergy,n_mels=n_mels)
        noiseLevel = 1.
        #Add randomness for levels 1.0, 2.0 (and 4.0?)
        whiteNoise = randGen.normal(scale=1,size=x.shape[0])
        whiteNoise = whiteNoise/np.max(np.abs(whiteNoise))*(noiseLevel/4)
        x = x + whiteNoise
        #Brown noise, implementation taken from:
        #https://github.com/python-acoustics/python-acoustics/blob/master/acoustics/generator.py, Last accessed: 16.04.2018, 18:09
        uneven = x.shape[0]%2
        randSpec = randGen.randn(x.shape[0]//2+1+uneven) + 1j * randGen.randn(x.shape[0]//2+1+uneven)
        specFilter = (np.arange(len(randSpec))+1)# Filter
        cutOff = 5 + randGen.rand()*10
        specFilter[0:int(len(randSpec)/self.rate*cutOff)] = len(randSpec)+1
        brownNoise = (np.fft.irfft(randSpec/specFilter)).real
        brownNoise = brownNoise/np.max(np.abs(brownNoise))*noiseLevel#/np.sqrt(np.mean(brownNoise**2))*(noiseLevel/1)
        x = x + brownNoise
        x = x * audioLevel
        x = melSpecExtraction(x,sr=self.rate,N_FFT=N_FFT,HOP=hop,center=True,log_power=log_power,deriv=deriv,frameEnergy=frameEnergy,n_mels=n_mels)
        y = y[::hop,:]
        return (x,y,y_ae)
    
    def estimateStatistics(self,data,numTestScenes,lengthScene=config_.lengthScene,randGen=np.random.RandomState(1337),eventsPerScene=config_.numEvents,N_FFT=config_.nFFT,hop=config_.hop,log_power=True,deriv=True,frameEnergy=True,n_mels=config_.nBin):
        initializationScene,_,_ = self.genScene(data,lengthScene=lengthScene,randGen=randGen,eventsPerScene=eventsPerScene,N_FFT=N_FFT,hop=hop,log_power=log_power,deriv=deriv,frameEnergy=frameEnergy,n_mels=n_mels)
        minimum = np.min(initializationScene,axis=0)
        maximum = np.max(initializationScene,axis=0)
        mean = np.mean(initializationScene,axis=0)
        var = np.var(initializationScene,axis=0)
        for i in range(numTestScenes):
            x,_,_ = self.genScene(data,lengthScene=lengthScene,randGen=randGen,eventsPerScene=eventsPerScene,N_FFT=N_FFT,hop=hop,log_power=log_power,deriv=deriv,frameEnergy=frameEnergy,n_mels=n_mels)
            minX = np.min(x,axis=0)
            minimum = np.where(minX<minimum,minX,minimum)
            maxX = np.max(x,axis=0)
            maximum = np.where(maxX>maximum,maxX,maximum)
            #As long as scenes are relative long this calculation should be okay
            #For small scenes calculation can become numerically unstable
            meanX = np.mean(x,axis=0)
            totalMean = (mean+meanX)/2.0
            var = ((var+mean-totalMean)+(np.var(x,axis=0)+meanX-totalMean))/2.0
            mean = totalMean
        std = np.sqrt(var)
        self.trainMax=maximum
        self.trainMin=minimum
        self.trainMean=mean
        self.trainStd=std
        return maximum,minimum,mean,std
                    
    
    def genTrainSetExample(self,data,lengthScene=config_.lengthScene,lengthExample=config_.lengthExample,
                           hopExample=config_.hopExample,asFeatureVector=False,temporalCompression=False,eventsPerScene=config_.numEvents,
                           N_FFT=config_.nFFT,hop=config_.hop,randSeed=1337,log_power=True,deriv=True,
                           frameEnergy=True,n_mels=config_.nBin,max_dB=120,predDelay=config_.predictionDelay,autoencoder=False,
                           zNorm=True,minMaxNorm=False,l2Norm=False):
        randomGenerator = np.random.RandomState(randSeed)
        cur_x, cur_y, cur_y_ae = self.genScene(data,lengthScene=lengthScene,randGen=randomGenerator,eventsPerScene=eventsPerScene,N_FFT=N_FFT,hop=hop,log_power=log_power,deriv=deriv,frameEnergy=frameEnergy,n_mels=n_mels,autoencoder=autoencoder)
        next_x, next_y, next_y_ae = self.genScene(data,lengthScene=lengthScene,randGen=randomGenerator,eventsPerScene=eventsPerScene,N_FFT=N_FFT,hop=hop,log_power=log_power,deriv=deriv,frameEnergy=frameEnergy,n_mels=n_mels,autoencoder=autoencoder)
        if zNorm:
            cur_x = cur_x - self.trainMean
            cur_x = cur_x / self.trainStd
            cur_y_ae = cur_y_ae - self.trainMean
            cur_y_ae = cur_y_ae / self.trainStd
            next_x = next_x - self.trainMean
            next_x = next_x / self.trainStd
            next_y_ae = next_y_ae - self.trainMean
            next_y_ae = next_y_ae / self.trainStd
        if minMaxNorm:
            cur_x = (cur_x - self.trainMin)/(self.trainMax-self.trainMin)
            cur_y_ae = (cur_y_ae - self.trainMin)/(self.trainMax-self.trainMin)
            next_x = (next_x - self.trainMin)/(self.trainMax-self.trainMin)
            next_y_ae = (next_y_ae - self.trainMin)/(self.trainMax-self.trainMin)
        if asFeatureVector:
            example_x = np.zeros((1,cur_x.shape[1]*lengthExample))
            example_y = np.zeros((1,cur_y.shape[1]))
            example_y_ae = np.zeros((1,cur_y_ae.shape[1]*lengthExample))
        else:
            example_x = np.zeros((lengthExample,cur_x.shape[1]))
            example_y = np.zeros((lengthExample,cur_y.shape[1]))
            example_y_ae = np.zeros((lengthExample,cur_y_ae.shape[1]))
        if temporalCompression:
            example_y = np.zeros((1,cur_y.shape[1]))
        else:
            example_y = np.zeros((lengthExample,cur_y.shape[1]))
        startExample = 0
        endExample = startExample+lengthExample
        while True:
            #If current scene has been fully used change next scene to current scene
            if startExample >= cur_x.shape[0]:
                startExample = startExample - cur_x.shape[0]
                endExample = startExample + lengthExample
                cur_x = next_x
                cur_y = next_y
                cur_y_ae = next_y_ae
                next_x, next_y, next_y_ae = self.genScene(data,lengthScene=lengthScene,randGen=randomGenerator,eventsPerScene=eventsPerScene,N_FFT=N_FFT,hop=hop,log_power=log_power,deriv=deriv,frameEnergy=frameEnergy,n_mels=n_mels,autoencoder=autoencoder)
                if zNorm:
                    next_x = next_x - self.trainMean
                    next_x = next_x / self.trainStd
                    next_y_ae = next_y_ae - self.trainMean
                    next_y_ae = next_y_ae / self.trainStd
                if minMaxNorm:
                    next_x = ((next_x - self.trainMin)/(self.trainMax-self.trainMin))*2-1
                    next_y_ae = ((next_y_ae - self.trainMin)/(self.trainMax-self.trainMin))*2-1
            #if the example extends over the end of the scene create a combination of both
            if endExample > cur_x.shape[0]:
                tempExample_x = np.concatenate((cur_x[startExample:],next_x[0:endExample-cur_x.shape[0]]),axis=0)
                tempExample_y = np.concatenate((cur_y[startExample:],next_y[0:endExample-cur_y.shape[0]]),axis=0)
                tempExample_y_ae = np.concatenate((cur_y_ae[startExample:],next_y_ae[0:endExample-cur_y_ae.shape[0]]),axis=0)
            else:
                tempExample_x = cur_x[startExample:endExample]
                tempExample_y = cur_y[startExample:endExample]
                tempExample_y_ae = cur_y_ae[startExample:endExample]
            if asFeatureVector:
                example_x = np.reshape(tempExample_x,example_x.shape)
                example_y_ae = np.reshape(tempExample_y_ae,example_y_ae.shape)
                #Sum the label
                #example_y = np.reshape(tempExample_y,example_y.shape)
                example_y = np.sum(tempExample_y,axis=0)
                example_y = np.where(example_y > lengthExample//2,1,0)
            else:
                example_x = tempExample_x
                example_y = tempExample_y
                example_y_ae = tempExample_y_ae
            if temporalCompression:
                example_y = np.sum(tempExample_y,axis=0)
                example_y = np.where(example_y > lengthExample/2,1,0)
            else:
                example_y = tempExample_y
            if l2Norm:
                example_x = example_x / np.sqrt(np.sum(example_x**2,axis=-1,keepdims=True))
                example_y_ae = example_y_ae / np.sqrt(np.sum(example_y_ae**2,axis=-1,keepdims=True))
            startExample += hopExample
            endExample = startExample + lengthExample
            yield (example_x,example_y,example_y_ae)
            
    def genTrainSetBatchOrdered(self,data,batchSize=config_.batchSize,lengthScene=config_.lengthScene,lengthExample=config_.lengthExample,
                           hopExample=config_.hopExample,asFeatureVector=False,temporalCompression=False,eventsPerScene=config_.numEvents,
                           N_FFT=config_.nFFT,hop=config_.hop,randSeed=1337,log_power=True,deriv=True,
                           frameEnergy=True,n_mels=config_.nBin,max_dB=120,predDelay=config_.predictionDelay,autoencoder=False,
                           zNorm=True,minMaxNorm=False,l2Norm=False):
        generators = []
        for example in range(batchSize):
            generators.append(self.genTrainSetExample(data,lengthScene=lengthScene,lengthExample=lengthExample,hopExample=hopExample,
                                                      asFeatureVector=asFeatureVector,temporalCompression=temporalCompression,
                                                      eventsPerScene=eventsPerScene,N_FFT=N_FFT,hop=hop,randSeed=randSeed+example,
                                                      log_power=log_power,deriv=deriv,frameEnergy=frameEnergy,n_mels=n_mels,
                                                      max_dB=max_dB,predDelay=predDelay,autoencoder=autoencoder,
                                                      zNorm=zNorm,minMaxNorm=minMaxNorm,l2Norm=l2Norm))
        testExample_x, testExample_y, testExample_y_ae = next(generators[0])
        if asFeatureVector:
            batchArray_x = np.zeros((batchSize,testExample_x.shape[-1]))
            batchArray_y = np.zeros((batchSize,testExample_y.shape[-1]))
            batchArray_y_ae = np.zeros((batchSize,testExample_y_ae.shape[-1]))
        else:
            batchArray_x = np.zeros((batchSize,testExample_x.shape[0],testExample_x.shape[1]))
            batchArray_y = np.zeros((batchSize,testExample_y.shape[0],testExample_y.shape[1]))
            batchArray_y_ae = np.zeros((batchSize,testExample_y_ae.shape[0],testExample_y_ae.shape[1]))
        while True:
            for num, gen in enumerate(generators):
                x,y,y_ae = next(gen)
                batchArray_x[num] = x
                batchArray_y[num] = y
                batchArray_y_ae[num] = y_ae
            if autoencoder:
                yield (batchArray_x,batchArray_y_ae)
            else:
                yield (batchArray_x,batchArray_y)
        
                
    def genTrainSetBatch(self,data,batchSize=config_.batchSize,lengthScene=config_.lengthScene,lengthExample=config_.lengthExample,
                           hopExample=config_.hopExample,asFeatureVector=False,temporalCompression=False,eventsPerScene=config_.numEvents,
                           N_FFT=config_.nFFT,hop=config_.hop,randSeed=1337,log_power=True,deriv=True,
                           frameEnergy=True,n_mels=config_.nBin,max_dB=120,predDelay=config_.predictionDelay,autoencoder=False,
                           zNorm=True,minMaxNorm=False,l2Norm=False):
        #Holds lists of all examples (shuffled) in generated scenes
        npRandom = np.random.RandomState(randSeed)
        scenes = deque(maxlen=batchSize*10)
        
        def sceneGenCallback(fileData):
            scenes.append(fileData)
        def errCallback(exception):
            print(exception)
        
        initialFill = []
        for i in range(batchSize*9):
            curData = []
            for i in range(eventsPerScene):
                curData.append(data[npRandom.randint(len(data))])
            seed = npRandom.randint(10000)
            initialFill.append((curData,seed,self.rate,lengthScene,len(self.labels),N_FFT,hop,log_power,deriv,frameEnergy,n_mels))
        with Pool() as pool:
            initialFill = pool.starmap(dcase2ProcessData,initialFill)
        for sceneNum in range(len(initialFill)):
            scene = initialFill.pop()
            examples = sceneToExamples(scene,lengthExample,hopExample,self.trainMean,self.trainStd,self.trainMeanTarget,self.trainStdTarget)
            scenes.append(examples)
        if asFeatureVector:
            batchX = np.zeros((batchSize,lengthExample*n_mels))
            batchY_AE = np.zeros((batchSize,lengthExample*n_mels))
        else:
            batchX = np.zeros((batchSize,lengthExample,n_mels))
            batchY_AE = np.zeros((batchSize,lengthExample,n_mels))
        batchY = np.zeros((batchSize,len(self.labels)))
        
        with Pool() as pool:
            while True:
                for exampleNum in range(batchSize):
                    #Wait for enough scenes to be ready
                    #TODO: actual wait with an availability object would be better
                    while len(scenes) <= batchSize*5:
                        time.sleep(2)
                    scene = scenes.popleft()
                    #Randomly skips a scene to create semi random batches
                    while npRandom.rand() < 0.5:
                        scenes.append(scene)
                        scene = scenes.popleft()
                    if asFeatureVector:
                        x = np.reshape(scene[0].pop(),(1,-1))
                        y_ae = np.reshape(scene[2].pop(),(1,-1))
                        batchX[exampleNum] = x
                        batchY_AE[exampleNum] = y_ae
                    else:
                        batchX[exampleNum] = scene[0].pop()                    
                        batchY_AE[exampleNum] = scene[2].pop()
                    batchY[exampleNum] = scene[1].pop()
                    if len(scene[0]) > 0:
                        scenes.append(scene)
                    else:
                        curData = []
                        for i in range(eventsPerScene):
                            curData.append(data[npRandom.randint(len(data))])
                        pool.apply_async(getSceneExamples,(curData,npRandom.randint(10000),
                                                                self.rate,lengthScene,len(self.labels),N_FFT,hop,
                                                                log_power,deriv,frameEnergy,n_mels,
                                                                lengthExample,hopExample,self.trainMean,
                                                                self.trainStd,self.trainMeanTarget,
                                                                self.trainStdTarget),callback=sceneGenCallback,error_callback=errCallback)
                if autoencoder:
                    yield (batchX,batchY_AE)
                else:
                    yield (batchX,batchY)
        
    def prepareValSet(self,lengthScene=config_.lengthScene,lengthExample=config_.lengthExample,
                      hopExample=config_.hopExample,asFeatureVector=False,temporalCompression=False,
                      eventsPerScene=config_.numEvents,N_FFT=config_.nFFT,hop=config_.hop,randSeed=1337,
                      log_power=True,deriv=True,frameEnergy=True,n_mels=config_.nBin,max_dB=120,
                      predDelay=config_.predictionDelay,relativeLevel=1.0,zNorm=True,minMaxNorm=False,l2Norm=False):
        data = self.devData
        totalLength = data[0][1].shape[0]
        testShape = melSpecExtraction(np.zeros((totalLength)),sr=self.rate,N_FFT=N_FFT,HOP=hop,center=True,log_power=log_power,deriv=deriv,frameEnergy=frameEnergy,n_mels=n_mels,max_dB=max_dB).shape
        x = np.zeros((len(data),testShape[0],testShape[1]))
        y = np.zeros((len(data),testShape[0],len(self.labels)))
        for index, example in enumerate(data):
            dataPoint = example[1]
            cur_x = melSpecExtraction(dataPoint,sr=self.rate,N_FFT=N_FFT,HOP=hop,center=True,log_power=log_power,deriv=deriv,frameEnergy=frameEnergy,n_mels=n_mels,max_dB=max_dB)
            cur_y = parseStringListToEventList(example[0],self.labels)
            cur_y = evaluation.eventListToRoll(cur_y,1./(self.rate/hop),testShape[0],len(self.labels))
            x[index] = cur_x
            y[index] = cur_y
        if zNorm:
            x = x - self.trainMean[np.newaxis]
            x = x / self.trainStd[np.newaxis]
        if minMaxNorm:
            x = ((x - self.trainMin)/(self.trainMax-self.trainMin))*2-1
        if asFeatureVector:
            print('Reshape to feature vector')
            paddingArray = np.zeros((x.shape[0],(int(lengthExample/2)),x.shape[-1]))
            paddingArray2 = np.zeros((x.shape[0],(int(lengthExample/2)),x.shape[-1]))
            x = np.concatenate((paddingArray,x,paddingArray2),axis=1)
            x = utils.sliding_window(x,lengthExample,hopExample,axis=1)            
            x = np.swapaxes(x, 2, 3)
            x = np.reshape(x,(-1,x.shape[-1]*lengthExample))
        if temporalCompression:
            x = np.reshape(x,(-1,x.shape[-1]))
            paddingArray = np.zeros(((int(lengthExample/2)),x.shape[-1]))
            x = np.concatenate((paddingArray,x,paddingArray),axis=0)
            x = utils.sliding_window(x,lengthExample,hopExample,axis=0)
            x = np.swapaxes(x,1,2)
            x = np.reshape(x,(-1,lengthExample,x.shape[-1]))
        if l2Norm:
            l2 = np.sum(x**2,axis=-1,keepdims=True)
            np.place(l2,l2==0,1e-12)
            l2 = np.sqrt(l2)
            x = x / l2
        print('ValData shape x {}, shape y {}'.format(x.shape,y.shape))
        return (x,y)
        
    def prepareTestSet(self,lengthScene=config_.lengthScene,lengthExample=config_.lengthExample,hopExample=config_.hopExample,asFeatureVector=False,temporalCompression=False,eventsPerScene=config_.numEvents,N_FFT=config_.nFFT,hop=config_.hop,randSeed=1337,log_power=True,deriv=True,frameEnergy=True,n_mels=config_.nBin,max_dB=120,predDelay=config_.predictionDelay,relativeLevel=1.0,zNorm=True,minMaxNorm=False,l2Norm=False):
        data = self.testData
        #Assumes that all test scenes have the same length
        totalLength = data[0][1].shape[0]
        testShape = melSpecExtraction(np.zeros((totalLength)),sr=self.rate,N_FFT=N_FFT,HOP=hop,center=True,log_power=log_power,deriv=deriv,frameEnergy=frameEnergy,n_mels=n_mels,max_dB=max_dB).shape
        x = np.zeros((len(data),testShape[0],testShape[1]))
        y = np.zeros((len(data),testShape[0],len(self.labels)))
        for index, example in enumerate(data):
            dataPoint = example[1]
            cur_x = melSpecExtraction(dataPoint,sr=self.rate,N_FFT=N_FFT,HOP=hop,center=True,log_power=log_power,deriv=deriv,frameEnergy=frameEnergy,n_mels=n_mels,max_dB=max_dB)
            cur_y = parseStringListToEventList(example[0],self.labels)
            cur_y = evaluation.eventListToRoll(cur_y,1/(self.rate/hop),testShape[0],len(self.labels))
            x[index] = cur_x
            y[index] = cur_y
        if zNorm:
            x = x - self.trainMean[np.newaxis]
            x = x / self.trainStd[np.newaxis]
        if minMaxNorm:
            x = ((x - self.trainMin)/(self.trainMax-self.trainMin))*2-1
        if asFeatureVector:
            print('Reshape to feature vector')
            x = np.reshape(x,(-1,x.shape[-1]))
            paddingArray = np.zeros(((int(lengthExample/1)),x.shape[-1]))
            x = np.concatenate((paddingArray,x),axis=0)
            x = utils.sliding_window(x,lengthExample,hopExample,axis=0)
            x = np.swapaxes(x, 1, 2)
            x = np.reshape(x,(-1,x.shape[2]*lengthExample))
        if temporalCompression:
            x = np.reshape(x,(-1,x.shape[-1]))
            paddingArray = np.zeros(((int(lengthExample/1)),x.shape[-1]))
            x = np.concatenate((paddingArray,x),axis=0)
            x = utils.sliding_window(x,lengthExample,hopExample,axis=0)
            x = np.swapaxes(x,1,2)
            x = np.reshape(x,(-1,lengthExample,x.shape[-1]))
        if l2Norm:
            l2 = np.sum(x**2,axis=-1,keepdims=True)
            np.place(l2,l2==0,1e-12)
            l2 = np.sqrt(l2)
            x = x / l2
        print('TestData shape x {}, shape y {}'.format(x.shape,y.shape))
        return (x,y)


#Outside of UrbanSound8K class, because we want to pass to multiprocessing pool
#and will have problems when it tries to pickle our object
def urbanSoundProcessExample(example,signalRate,lengthScene,asFeatureVector,temporalCompression,N_FFT,
                   hop,log_power,deriv,frameEnergy,n_mels):
        
    #Take example with label (and params about feature extraction)
    #cut example into equal length pieces (with zero padding for last)
    #apply mel-spec extraction, reshape as necessary
    #note that we want to use length example here potentially more as seconds
    #each subsection will also probably not overlap
    #return list of all pieces feature vectors and label vectors
    
    label = example[0]
    labelVector = np.zeros(10)
    labelVector[label] = 1
    data = example[1]
    lengthInSamples = int(lengthScene*signalRate)
    maxParts = int(np.ceil(len(data)/lengthInSamples*1.0))
    data = np.append(data,np.zeros(maxParts*lengthInSamples-len(data)))
    data = np.reshape(data,(maxParts,lengthInSamples))
    allParts = []
    for row in data:
        melSpec = melSpecExtraction(row,sr=signalRate,N_FFT=N_FFT,HOP=hop,log_power=log_power,
                                    deriv=deriv,frameEnergy=frameEnergy,n_mels=n_mels)
        if asFeatureVector:
            melSpec = np.reshape(melSpec,(1,-1))
        allParts.append((melSpec,labelVector))
    return allParts
    
def urbanSoundProcessExampleNoSegmentation(example,signalRate,N_FFT,hop,log_power,
                                           deriv,frameEnergy,n_mels):
    label = example[0]
    labelVector = np.zeros(10)
    labelVector[label] = 1
    data = example[1]
    melSpec = melSpecExtraction(data,sr=signalRate,N_FFT=N_FFT,HOP=hop,log_power=log_power,
                                deriv=deriv,frameEnergy=frameEnergy,n_mels=n_mels)
    return (melSpec,labelVector)
        
class UrbanSound8KdataPrep(dataPrep):
    baseFolder = config_.UrbanSound8KBaseFolder
    
    def __init__(self, SR=16000,dev=True,test=True,stats=None):
        self.rate = SR
        self.folds = []
        for i in range(10):
            self.folds.append(loadDataUrbanSound8K(UrbanSound8KdataPrep.baseFolder,sampling_rate=SR,fold=i+1))
        #TODO: dynamically arange folds into train, dev and test data
        self.labels = self.folds[0][1]
        self.trainData = []
        for i in range(8):
            self.trainData.extend(self.folds[i][0])
        self.devData=None
        if dev:
            self.devData = self.folds[8][0]
        self.testData=None
        if test:
            self.testData = self.folds[9][0]
        self.holdoutData = self.devData
        if stats:
            self.trainMean, self.trainStd, self.trainMin, self.trainMax = stats
        else:
            self.trainMean = None
            self.trainStd = None
            self.trainMin = None
            self.trainMax = None
        
    def processDataToArray(self,data,lengthScene,asFeatureVector,temporalCompression,N_FFT,hop,
                           log_power,deriv,frameEnergy,n_mels,evaluationData=False):
        #We can use parallel processing here
        #So we need an extra function taking an example with label
        featureSpecificProcessFunc = partial(urbanSoundProcessExample,signalRate=self.rate,lengthScene=lengthScene,asFeatureVector=asFeatureVector,
                                             temporalCompression=temporalCompression,N_FFT=N_FFT,hop=hop,
                                             log_power=log_power,deriv=deriv,frameEnergy=frameEnergy,n_mels=n_mels)
        with Pool() as pool:
            processedData = pool.map(featureSpecificProcessFunc,data)
        xList = []
        yList = []
        #We need special treatment for evaluation data, as we need to evaluate on full examples and not parts
        #As such we mark sure that every example gets the same length with fakeExamples
        for example in processedData:
            if evaluationData:
                maxParts = int(np.ceil(4. / lengthScene))
                #Fake data might mess with plots but is easy to spot as labels are all one
                appendedFakeData = [(np.zeros_like(example[0][0]),np.ones_like(example[0][1])) for i in range(maxParts-len(example))]
                example.extend(appendedFakeData)
                assert len(example) == maxParts, 'example is of len {}, fake data is of len {}, max parts is {}'.format(len(example),len(appendedFakeData),maxParts)
            for part in example:
                xList.append(part[0])
                yList.append(part[1])
        x = np.array(xList)
        y = np.array(yList)
        return (x,y)
    
    def normaliseData(self,data,zNorm,minMaxNorm,l2Norm):
        if zNorm and minMaxNorm:
            print('''Warning: using both zNorm and minMaxNorm is not useful and 
                  the implementation will ignore the minMaxNorm''')
        if zNorm:
            data = data - self.trainMean
            data = data / self.trainStd
        elif minMaxNorm:
            data = ((data - self.trainMin)/(self.trainMax-self.trainMin))*2-1
        if l2Norm:
            l2 = np.sum(data**2,axis=-1,keepdims=True)
            l2 = np.where(l2==0,1e-12,l2)
            l2 = np.sqrt(l2)
            data = data / l2
        return data
        
    def prepareTrainSet(self,data,lengthScene=config_.lengthScene,lengthExample=config_.lengthExample,
                        hopExample=config_.hopExample,asFeatureVector=False,temporalCompression=False,
                        eventsPerScene=config_.numEvents,N_FFT=config_.nFFT,hop=config_.hop,predDelay=0,
                        randSeed=1337,log_power=True,deriv=True,frameEnergy=True,n_mels=config_.nBin,
                        shuffle=True,zNorm=True,minMaxNorm=False,l2Norm=False):
        
        x, y = self.processDataToArray(data,lengthScene,asFeatureVector,temporalCompression,
                                       N_FFT,hop,log_power,deriv,frameEnergy,n_mels)
        if self.trainMean is None:
            self.trainMean = np.mean(x,axis=(0,1),keepdims=True)
        if self.trainStd is None:
            self.trainStd = np.std(x,axis=(0,1),keepdims=True)
        if self.trainMin is None:
            self.trainMin = np.min(x,axis=(0,1),keepdims=True)
        if self.trainMax is None:
            self.trainMax = np.max(x,axis=(0,1),keepdims=True)
        x = self.normaliseData(x,zNorm,minMaxNorm,l2Norm)
        if asFeatureVector:
            x = np.squeeze(x) 
        y_ae = x
        if shuffle:
            randGen = np.random.RandomState(randSeed)
            randIndex = np.arange(x.shape[0])
            randGen.shuffle(randIndex)
            x = x[randIndex]
            y = y[randIndex]
            y_ae = y_ae[randIndex]
        print('X shape {} y shape {}'.format(x.shape,y.shape))
        return (x,y,y_ae)
        
    def genTrainSetBatch(self,data,batchSize=config_.batchSize,lengthScene=config_.lengthScene,lengthExample=config_.lengthExample,
                           hopExample=config_.hopExample,asFeatureVector=False,temporalCompression=False,eventsPerScene=config_.numEvents,
                           N_FFT=config_.nFFT,hop=config_.hop,randSeed=1337,log_power=True,deriv=True,
                           frameEnergy=True,n_mels=config_.nBin,max_dB=120,predDelay=config_.predictionDelay,autoencoder=False,
                           zNorm=True,minMaxNorm=False,l2Norm=False):

        featureSpecificProcessFunc = partial(urbanSoundProcessExampleNoSegmentation,signalRate=self.rate,
                                             N_FFT=N_FFT,hop=hop,log_power=log_power,
                                             deriv=deriv,frameEnergy=frameEnergy,n_mels=n_mels)
        print('Prepare data for generator')
        with Pool() as pool:
            preProcessedData = pool.map(featureSpecificProcessFunc,data)
        testShape = melSpecExtraction(np.zeros(int(self.rate*lengthScene)),sr=self.rate,N_FFT=N_FFT,HOP=hop,log_power=log_power,
                                deriv=deriv,frameEnergy=frameEnergy,n_mels=n_mels).shape
        for num, example in enumerate(preProcessedData):
            exampleX, exampleY = example
            exampleX = self.normaliseData(exampleX[np.newaxis,:,:],zNorm,minMaxNorm,l2Norm)
            if exampleX.shape[1] <= testShape[0]:
                exampleX = np.tile(exampleX,(1,int(np.ceil(testShape[0]*1.0/exampleX.shape[1]*1.)),1))
            preProcessedData[num] = (exampleX,exampleY)
        if asFeatureVector:
            testShape = (1,testShape[0]*testShape[1])
        batchArray_x = np.zeros((batchSize,testShape[0],testShape[1]))
        batchArray_y = np.zeros((batchSize,len(self.labels)))
        batchArray_y_ae = np.zeros_like(batchArray_x)
        randGen = np.random.RandomState(randSeed)
        print('Generator ready')
        while True:
            for i in range(batchArray_x.shape[0]):
                example = preProcessedData[randGen.randint(len(preProcessedData))]
                startPoint = randGen.randint(example[0].shape[1]-testShape[0]+1)
                endPoint = startPoint+testShape[0]
                snippet = example[0][:,startPoint:endPoint,:]
                batchArray_x[i] = snippet
                batchArray_y[i] = example[1]
                batchArray_y_ae[i] = snippet #Manipulate snippet if target of ae should differ from input
            if autoencoder:
                yield (batchArray_x,batchArray_y_ae)
            else:
                yield (batchArray_x,batchArray_y)
    
    def prepareValSet(self,lengthScene=config_.lengthScene,lengthExample=config_.lengthExample,
                        hopExample=config_.hopExample,asFeatureVector=False,temporalCompression=False,
                        eventsPerScene=config_.numEvents,N_FFT=config_.nFFT,hop=config_.hop,predDelay=0,
                        randSeed=1337,log_power=True,deriv=True,frameEnergy=True,n_mels=config_.nBin,
                        zNorm=True,minMaxNorm=False,l2Norm=False):
        data = self.devData
        x, y= self.processDataToArray(data,lengthScene,asFeatureVector,temporalCompression,
                                      N_FFT,hop,log_power,deriv,frameEnergy,n_mels,evaluationData=True)
        x = self.normaliseData(x,zNorm,minMaxNorm,l2Norm)
        if asFeatureVector:
            x = np.squeeze(x)
        print('ValData shape x {}, shape y {}'.format(x.shape,y.shape))
        return (x,y)
    
    def prepareTestSet(self,lengthScene=config_.lengthScene,lengthExample=config_.lengthExample,
                        hopExample=config_.hopExample,asFeatureVector=False,temporalCompression=False,
                        eventsPerScene=config_.numEvents,N_FFT=config_.nFFT,hop=config_.hop,predDelay=0,
                        randSeed=1337,log_power=True,deriv=True,frameEnergy=True,n_mels=config_.nBin,
                        zNorm=True,minMaxNorm=False,l2Norm=False):
        data = self.testData
        x, y= self.processDataToArray(data,lengthScene,asFeatureVector,temporalCompression,
                                      N_FFT,hop,log_power,deriv,frameEnergy,n_mels,evaluationData=True)
        x = self.normaliseData(x,zNorm,minMaxNorm,l2Norm)
        if asFeatureVector:
            x = np.squeeze(x)
        return (x,y)
        
    
if __name__ == "__main__":
    loader = DCASE2dataPrep(SR=30000,dev=False,test=False)
    print(len(loader.trainData))
    print(len(loader.holdoutData))
    signal_rate = loader.rate
    minTimeStretch, maxTimeStretch, stepsTimeStretch, minPitchShift, maxPitchShift, stepsPitchShift, minLevelAdjust, maxLevelAdjust, stepsLevelAdjust = config_.getAugmentConfig(config=5)
    print('Augment')
    aug = augment(loader.trainData,signal_rate,minTimeStretch,maxTimeStretch,
                  stepsTimeStretch,minPitchShift,maxPitchShift,stepsPitchShift,
                  minLevelAdjust,maxLevelAdjust,stepsLevelAdjust)
    nBin = 200
    print('Prepare Train Set')
    trainSet = loader.prepareTrainSet(aug,deriv=False,frameEnergy=False,N_FFT=2048,hop=512,n_mels=nBin,lengthScene=20,eventsPerScene=5,lengthExample=500,hopExample=500,asFeatureVector=False,shuffle=False)
    train_x, train_y, train_y_ae = trainSet
    numExamples = 1
    print('Plot train')
    #length = 500
    #train_x = np.resize(train_x,(train_x.shape[0]//length,length,train_x.shape[1]))
    #train_y = np.resize(train_y,(train_y.shape[0]//length,length,train_y.shape[1]))
    utils.plotAsPcolor(train_x,'plots/test',numExamples)
    print('Plot train target')
    utils.plotAsPcolor(train_y,'plots/testTarget',numExamples,ylabel='Class') 

