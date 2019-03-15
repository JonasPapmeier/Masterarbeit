# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 14:43:24 2018

@author: 1papmeie
"""

import numpy as np
import os
import multiprocessing
import time
from collections import deque
import audioProcessing
from functools import partial
import modelLoader
import dataLoader
import random
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TerminateOnNaN, CSVLogger
import config_
#import utils


def getFileList(baseFolder,signalRate):
    folders = os.listdir(baseFolder)
    fileList = []
    for folder in folders:
        #For later usage of audioProcessing.loadSingleFileAudio
        descriptionTuples = list(map(lambda fileName: (baseFolder+folder+'/',fileName,signalRate),os.listdir(baseFolder+folder)))
        fileList.extend(descriptionTuples)
    return fileList
        
def processFile(fileData,signalRate,lengthExample,asFeatureVector,temporalCompression,N_FFT,
                   hop,log_power,deriv,frameEnergy,n_mels,zNorm):
        
    #Take example with label (and params about feature extraction)
    #cut example into equal length pieces (with zero padding for last)
    #apply mel-spec extraction, reshape as necessary
    #note that we want to use length example here potentially more as seconds
    #subsections will not overlap
    #return list of all pieces feature vectors
    """
    gotFile = False
    while not gotFile:
        try:
            print('Bla',fileDeque)
            fileData = fileDeque.popleft()
            gotFile = True
        except IndexError:
            time.sleep(1)
            print('Wait for file')
    """
    
    data = fileData[1]
    #data = data/np.max(np.abs(data))
    #data = data * np.random.random() * ((1./2.**5 - 1./2.**9) + 1./2.**9)
    lengthInSamples = int(lengthExample*hop-1)
    maxParts = int(np.ceil(len(data)/lengthInSamples*1.0))
    data = np.append(data,np.zeros(maxParts*lengthInSamples-len(data)))
    #data.resize((int(np.ceil(len(data)/lengthInSamples*1.0)),lengthInSamples))
    data = np.resize(data,(maxParts,lengthInSamples))
    processedParts = []
    for row in data:
        melSpec = audioProcessing.melSpecExtraction(row,sr=signalRate,N_FFT=N_FFT,HOP=hop,log_power=log_power,
                                    deriv=deriv,frameEnergy=frameEnergy,n_mels=n_mels)
        melSpec = melSpec - zNorm[0]
        melSpec = melSpec / zNorm[1]
        if asFeatureVector:
            melSpec = np.reshape(melSpec,(1,-1))
        processedParts.append(np.squeeze(melSpec))
    return processedParts
    
def processFileWhole(fileData,signalRate,lengthExample,asFeatureVector,temporalCompression,N_FFT,
                   hop,log_power,deriv,frameEnergy,n_mels,zNorm):
    data = fileData[1]
    melSpec = audioProcessing.melSpecExtraction(data,sr=signalRate,N_FFT=N_FFT,HOP=hop,log_power=log_power,
                                    deriv=deriv,frameEnergy=frameEnergy,n_mels=n_mels)
    melSpec = melSpec - zNorm[0]
    melSpec = melSpec / zNorm[1]
    #np.squeeze(melSpec)
    return melSpec
    
def loadAll(folder,batchSize,signalRate,lengthExample,asFeatureVector,temporalCompression,N_FFT,
                    hop,log_power,deriv,frameEnergy,n_mels,zNorm):
    allFiles = getFileList(folder,signalRate)    
    #allFiles = allFiles[0:10]
    featExtractionFunc = partial(processFileWhole,signalRate=signalRate,lengthExample=lengthExample,
                                 asFeatureVector=asFeatureVector,temporalCompression=temporalCompression,
                                 N_FFT=N_FFT,hop=hop,log_power=log_power,deriv=deriv,
                                 frameEnergy=frameEnergy,n_mels=n_mels,zNorm=zNorm)
    with multiprocessing.Pool() as pool:
        print('Load files')
        rawFiles = pool.map(audioProcessing.loadSingleFileAudio,allFiles)
        print('Extract features')
        processedFiles = pool.map(featExtractionFunc,rawFiles)
    exampleList = []
    print('build list')
    for fileData in processedFiles:
        exampleList.append(fileData)
    #allData = np.concatenate(exampleList,axis=0)
    return exampleList
    
def genBatchPreLoadWhole(folder,batchSize,signalRate,lengthExample,asFeatureVector,temporalCompression,N_FFT,
             hop,log_power,deriv,frameEnergy,n_mels,zNorm):
    nonSegmentedFiles = loadAll(folder,batchSize,signalRate,lengthExample,asFeatureVector,temporalCompression,N_FFT,
                    hop,log_power,deriv,frameEnergy,n_mels,zNorm)
    testShape = nonSegmentedFiles[0].shape
    print(len(nonSegmentedFiles), testShape)
    randGen = np.random.RandomState(1337)
    batchArray_x = np.zeros((batchSize,lengthExample,testShape[-1]))
    while True:
        exampleIndexes = randGen.randint(len(nonSegmentedFiles),size=batchSize)
        #batchArray_x = np.zeros((batchSize,lengthExample,testShape[-1]))
        for i in range(batchSize):
            example = nonSegmentedFiles[exampleIndexes[i]]
            startIndex = randGen.randint(example.shape[0]-lengthExample)
            endIndex = startIndex+lengthExample
            segment = example[startIndex:endIndex]
            batchArray_x[i] = segment
        yield (batchArray_x,batchArray_x)
    
def genBatch(folder,batchSize,signalRate,lengthExample,asFeatureVector,temporalCompression,N_FFT,
             hop,log_power,deriv,frameEnergy,n_mels,zNorm):
    #TODO: fixed random generator?
    allFiles = getFileList(folder,signalRate)
    minFilesReady = batchSize*1
    usedFiles = []
    readyFiles = deque(maxlen=minFilesReady*4)
    rawFiles = deque(maxlen=minFilesReady*4)
    repeatExamples = deque(maxlen=5000)
    #preparationStuff
    featExtractionFunc = partial(processFile,signalRate=signalRate,lengthExample=lengthExample,
                                 asFeatureVector=asFeatureVector,temporalCompression=temporalCompression,
                                 N_FFT=N_FFT,hop=hop,log_power=log_power,deriv=deriv,
                                 frameEnergy=frameEnergy,n_mels=n_mels,zNorm=zNorm)
    def rawFileCallback(fileData):
        rawFiles.append(fileData)
    def readyFileCallback(fileData):
        readyFiles.append(fileData)
    def errCallback(exception):
        print(exception)
    #start up filling ready files once
    iterationCycle = 0
    with multiprocessing.Pool() as pool:
        initialFill = []
        for i in range(100):
            initialFill.append(allFiles[np.random.choice(len(allFiles))])
        initialFill = pool.map(audioProcessing.loadSingleFileAudio,initialFill)
        initialFill = pool.map(featExtractionFunc,initialFill)
        for initFile in initialFill:
            for example in initFile:
                repeatExamples.append(example)
        initialFill.clear()
        while True:
            if len(readyFiles) < minFilesReady-len(rawFiles):
                for i in range((minFilesReady-len(readyFiles))*1):
                    newFile = allFiles[np.random.choice(len(allFiles))]
                    pool.apply_async(audioProcessing.loadSingleFileAudio,(newFile,),callback=rawFileCallback,error_callback=errCallback)
            while len(rawFiles)<minFilesReady/2 and len(readyFiles)>minFilesReady:
                try:
                    rawFile = rawFiles.popleft()
                    pool.apply_async(featExtractionFunc,(rawFile,),callback=readyFileCallback,error_callback=errCallback)
                except IndexError:
                    break
            #Get up to batch size examples from used files
            batchExamples = []
            #TODO: maybe shuffle usedFiles regulary
            #print(usedFiles)
            if iterationCycle == 0:
                for i in reversed(range(len(usedFiles))):
                    example = usedFiles[i].pop()
                    if len(usedFiles[i]) == 0:
                        usedFiles.pop(i)
                    batchExamples.append(example)
                    repeatExamples.append(example)
                    if len(batchExamples) == batchSize:
                        random.shuffle(usedFiles)
                        break
                for j in range(batchSize-len(batchExamples)):
                    gotFile = False
                    while not gotFile:
                        try:
                            newFile = readyFiles.popleft()
                            gotFile = True
                        except IndexError:
                            #print('Have to wait for files')
                            time.sleep(1)
                            #while True:
                            try:
                                rawFile = rawFiles.popleft()
                                pool.apply_async(featExtractionFunc,(rawFile,),callback=readyFileCallback,error_callback=errCallback)
                            except IndexError:
                                pass
                            #        break
                    example = newFile.pop()
                    batchExamples.append(example)
                    repeatExamples.append(example)
                    if len(newFile) > 0:
                        usedFiles.append(newFile)
                print(len(readyFiles),len(rawFiles),len(repeatExamples))
            else:
                for k in range(batchSize):
                    example = repeatExamples[np.random.choice(len(repeatExamples))]
                    batchExamples.append(example)
            iterationCycle += 1
            iterationCycle = iterationCycle%((len(repeatExamples)//100)+1)
            batchArray_x = np.array(batchExamples)
            yield (batchArray_x,batchArray_x)

def main():
    randGen = np.random.RandomState(1337)     
    baseFolder = 'ESC/ESC-US/'
     
    signalRate=30000
    lengthScene = 10
    eventsPerScene = 5
    numberMelBins = 80
    
    nfft = 2048
    hop = 512
    layer = 2
    extractionLayer = 5
    hidden = 80
    lr = 0.002
    sigma = 0.25
    log_power =  True
    deriv = False
    frameEnergy = False
    
    batchSize=128
    epochs=100000
    
    asFeatureVector=False
    temporalCompression=True
    
    loader = dataLoader.DCASE2dataPrep(SR=signalRate,dev=False,test=False)
    trainData = loader.trainData
    
    #TODO: also get the augmented trainData
    print('augment train data')
    augmentedTrainData = audioProcessing.augment(trainData,signalRate,0.9,1.1,2,-1,1,2,0,3,4)
    lengthExample = 11
    hopExample=11
    #to compute a mean and std deviation for normalization
    print('prepare train set')
    trainSet = loader.prepareTrainSet(augmentedTrainData,lengthScene=lengthScene,lengthExample=lengthExample,hopExample=hopExample,
                                      eventsPerScene=eventsPerScene,N_FFT=nfft,hop=hop,randSeed=1337,
                                      log_power=log_power,deriv=deriv,frameEnergy=frameEnergy,asFeatureVector=asFeatureVector,temporalCompression=temporalCompression,
                                      n_mels=numberMelBins,predDelay=0)
    val = trainSet[0][0:10000]
                                         
    zNorm = (loader.trainMean,loader.trainStd)
    fullStats = (loader.trainMean,loader.trainStd,loader.trainMin,loader.trainMax)
    modelName = 'ESC_PreTrainNew6'
    #1 normal full
    #2 lower lr
    #3 no "noise" classes, Insects, rain, sea waves, crackling fire, crickets,
    #chirping birds, water drops, wind, pouring water, toilet flush, thunderstorm, clapping,
    #breathing, brushing teeth, drinking sipping, washing machine, vacuum cleaner,
    #helicopter, chainsaw, engine, train, airplane, fireworks, hand saw
    #4 no "noise" classes, also remove choughing, laughing, door knock, keyboard typing
    #5 new training variant with loading full file and chopping subsets
    #6 0.25 sigma dropout on input
    modelsFolder = config_.modelsFolder
    logFolder = config_.logFolder
    
    statsFileName = modelsFolder + 'ST_' + modelName
    print('Saving stats',statsFileName)
    np.savez(statsFileName,mean=fullStats[0],std=fullStats[1],minimum=fullStats[2],maximum=fullStats[3])
    
    lengthExample=11
    print('Create Generator')
    generator = genBatchPreLoadWhole(baseFolder,batchSize,signalRate,lengthExample,asFeatureVector,
                         temporalCompression,nfft,hop,log_power,deriv,frameEnergy,numberMelBins,zNorm)
    print('Create initial fill')
    testExample = next(generator)
    
    #dataX = loadAll(baseFolder,batchSize,signalRate,lengthExample,asFeatureVector,
    #                     temporalCompression,nfft,hop,log_power,deriv,frameEnergy,numberMelBins,zNorm)
    #testExample = dataX[0:1]
    #print(dataX.shape)
    
    inDim = testExample[0].shape #(1,lengthExample* (numberMelBins+1))
    outDim = inDim
    #model = modelLoader.createFeedForward_AE(layer,hidden,inDim,outDim,lr,sigma)
    model = modelLoader.createSeqToSeq(layer,hidden,inDim,outDim,lr,sigma)
    
    saveCallback = ModelCheckpoint(modelsFolder+modelName,monitor='val_loss',save_best_only=True)
    breakCallback = TerminateOnNaN()
    lrCallback = ReduceLROnPlateau(monitor='loss',factor=0.1,patience=100)
    logCallback = CSVLogger(logFolder+modelName,append=True)
    history = model.fit_generator(generator,steps_per_epoch=200,epochs=epochs,verbose=2,max_queue_size=30,callbacks=[saveCallback,breakCallback,lrCallback,logCallback],validation_data=(val,val),workers=0)
    #history = model.fit(x=dataX,y=dataX,batch_size=batchSize,epochs=epochs,verbose=2,callbacks=[saveCallback,breakCallback,lrCallback,logCallback],validation_data=(val,val))
    exit(0)

if __name__ == '__main__':
    exit(main())
