# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 18:12:35 2018

@author: 1papmeie
"""

import numpy as np
import keras.backend as K
from keras.models import load_model
from keras.callbacks import TensorBoard
import dataLoader
import utils
import audioProcessing
import modelLoader

if __name__ == '__main__':
    randGen = np.random.RandomState(1337)
    #"""
    loader = dataLoader.DCASE2dataPrep(SR=30000,test=False)
    trainData = loader.trainData
    byLabel = loader.splitDataByLabel(trainData,loader.labels)
    reducedTrainData = []
    for label in byLabel:
        for i in range(3):
            reducedTrainData.append(label.pop())
    trainData = reducedTrainData
    reducedValData = []
    valData = loader.devData
    randGen.shuffle(valData)
    for i in range(3):
        reducedValData.append(valData.pop())
    valData = reducedValData
    loader.devData = valData
    #"""
    
    #Augmentation
    signal_rate = loader.rate
    minTimeStretch = 0.9
    maxTimeStretch = 1.1
    stepsTimeStretch = 0
    minPitchShift = -1
    maxPitchShift = 1.1
    stepsPitchShift = 0
    minLevelAdjust = 0.0
    maxLevelAdjust = 2.0
    stepsLevelAdjust = 3
    augmentedTrainData = audioProcessing.augment(trainData,signal_rate,minTimeStretch,maxTimeStretch,stepsTimeStretch,minPitchShift,maxPitchShift,stepsPitchShift,minLevelAdjust,maxLevelAdjust,stepsLevelAdjust)
    
    lengthScene = 15
    eventsPerScene = 5
    numberMelBins = 81

    nfft = 2048
    hop = 512
    lengthExample=40 #40 should now be roughly a second input
    hopExample=20
    layer = 2
    extractionLayer = 4
    hidden = 80
    lr = 0.002
    sigma = 0.0
    asFeatureVector=False
    temporalCompression=False
    if asFeatureVector:
        inDim = (1,lengthExample* (numberMelBins+0))
    elif temporalCompression:
        inDim = (1,lengthExample, (numberMelBins+0))
    else:
        print('What are you trying to do?')
        inDim = (1,lengthExample,numberMelBins+0)
    outDim = inDim
    
    batchSize=128
    epochs=121
    
    #model = modelLoader.createFeedForward_AE(layer,hidden,inDim,outDim,lr,sigma)        
    #model = modelLoader.createSeqToSeq(layer,hidden,inDim,outDim,lr,sigma)
    model = modelLoader.createRNN_AE(layer,hidden,inDim,outDim,lr,sigma)
    #model = load_model('testSave')
    
    trainSet = loader.prepareTrainSet(augmentedTrainData,lengthScene=lengthScene,lengthExample=lengthExample,hopExample=hopExample,
                                      eventsPerScene=eventsPerScene,N_FFT=nfft,hop=hop,randSeed=1337,
                                      log_power=True,deriv=False,frameEnergy=False,asFeatureVector=asFeatureVector,temporalCompression=temporalCompression,
                                      n_mels=numberMelBins,predDelay=0,zNorm=True)
    lengthExample=10
    hopExample=4
    validationSet = loader.prepareValSet(lengthScene=lengthScene,lengthExample=lengthExample,hopExample=hopExample,eventsPerScene=eventsPerScene,
                                         N_FFT=nfft,hop=hop,randSeed=1337,log_power=True,asFeatureVector=asFeatureVector,temporalCompression=temporalCompression,
                                         deriv=False,frameEnergy=False,n_mels=numberMelBins,
                                         predDelay=0,zNorm=True)
                                         
    #trainSet = None
    #print(trainSet[0][0])
    #print(trainSet[2][0])
    #print(trainSet[0][0]-trainSet[2][0])
    #numEmbed=1000
    #tens = TensorBoard(log_dir='tensorboard/'+str(numberMelBins)+'Bin_'+str(hidden)+'hid_'+str(nfft)+'fft',
    #                   histogram_freq=5, batch_size=batchSize, write_graph=False, write_grads=True, write_images=False,
    #                   embeddings_freq=10,embeddings_layer_names=['dense_1'],embeddings_data=trainSet[0][0:numEmbed,...],embeddings_metadata='../../embeddings/test.tsv')
    #model.fit(x=trainSet[0],y=trainSet[2],batch_size=batchSize,epochs=epochs,verbose=2,callbacks=[tens],validation_data=(validationSet[0],validationSet[0]))
    model.fit(x=trainSet[0],y=trainSet[2],batch_size=batchSize,epochs=epochs,verbose=2,validation_data=(validationSet[0],validationSet[0]))    
    #with open('tensorboard/'+str(numberMelBins)+'Bin_'+str(hidden)+'hid_'+str(nfft)+'fft'+'/embeddings/test.tsv', 'w') as f:
    #with open('embeddings/test.tsv', 'w') as f:
    #    for i in range(numEmbed):
    #        pointLabel = np.concatenate((np.zeros(1),trainSet[1][i]),axis=-1)
    #        pointLabel = np.argmax(pointLabel,axis=-1)
    #        f.write('{}\n'.format(pointLabel))
    
    """        
    numEmbed=min(2000,train_x.shape[0])
        with open('embeddings/test.tsv', 'w') as f:
            for i in range(numEmbed):
                #pointLabel = np.concatenate((np.zeros(1),trainSet[1][i]),axis=-1)
                pointLabel = train_y[i]
                pointLabel = np.argmax(pointLabel,axis=-1)
                f.write('{}\n'.format(pointLabel))
        #tensCallback = TensorBoard(log_dir='tensorboard/'+self.modelName, histogram_freq=5, batch_size=self.batch_size, write_graph=False, write_grads=True, write_images=False)
        tensCallback = TensorBoard(log_dir='tensorboard/'+self.modelName,
                       histogram_freq=5, batch_size=self.batch_size, write_graph=False, write_grads=True, write_images=False,
                       embeddings_freq=300,embeddings_layer_names=['dense_1'],embeddings_data=train_x[0:numEmbed,...],embeddings_metadata='../../embeddings/test.tsv')
    """
    #generator = loader.genTrainSetBatchUnordered(augmentedTrainData,batchSize=batchSize,lengthScene=lengthScene,
    #                                             lengthExample=lengthExample,hopExample=hopExample,asFeatureVector=asFeatureVector,
    #                                             temporalCompression=temporalCompression,eventsPerScene=eventsPerScene,
    #                                             N_FFT=nfft,hop=hop,randSeed=1337,log_power=True,deriv=False,frameEnergy=False,
    #                                             n_mels=numberMelBins,predDelay=0,autoencoder=True,zNorm=True)
    #model.fit_generator(generator,steps_per_epoch=750//batchSize,epochs=epochs,verbose=2,max_queue_size=30,callbacks=[tens],validation_data=(validationSet[0],validationSet[0]),workers=0)
    
    #model.save('testSave')
    #model = load_model('testSave')
    #model.fit(x=trainSet[0],y=trainSet[2],batch_size=batchSize,epochs=epochs,verbose=2)
    
    input_tensor = model.layers[0].input
    layer_output = model.layers[extractionLayer].output
    extractionFunction = K.function([input_tensor,K.learning_phase()],[layer_output])
    transformedData = extractionFunction([validationSet[0],0])[0]
    #transformedData = extractionFunction([trainSet[0],0])[0]
    #skips=4
    #transformedData = transformedData[::skips,:]
    #fakeTestData = np.ones((3,3,81))
    #transFake = extractionFunction([fakeTestData,0])[0]
    #transFake2 = extractionFunction([fakeTestData,0])[0]
    #print(transFake)
    #print(transFake2)
    #exit()
    
    val_x = transformedData
    val_x = np.reshape(val_x,(-1,val_x.shape[-1]))
    print(val_x.shape)
    val_x = val_x[::hopExample,:]
    print(val_x.shape)
    #val_x = np.resize(val_x,(val_x.shape[0]//hopExample,hopExample,val_x.shape[-1]))
    #val_x = np.max(val_x,axis=1)
    #val_x = validationSet[0]
    val_y = validationSet[1]
    #train_y = trainSet[1]
    #val_y = trainSet[1]
    val_y = np.reshape(val_y,(-1,val_y.shape[-1]))
    val_y = val_y[::hopExample,:]
    print('val_y shape',val_y.shape)
    #val_y = np.resize(val_y,(val_y.shape[0]//hopExample,hopExample,val_y.shape[-1]))
    print('val_y shape',val_y.shape)
    #val_y = np.max(val_y,axis=1)
    print('val_y shape',val_y.shape)
    val_y = np.reshape(val_y,(-1,val_y.shape[-1]))
    #val_y = np.concatenate((val_y,train_y),axis=0)
    print('val_y shape',val_y.shape)
    #val_y = val_y[::hopExample,:]
    #val_y = val_y[hopExample-1::hopExample,:]
    #val_y = val_y[hopExample//2::hopExample,:]
    actualClassData = np.any(val_y,axis=-1)
    print('Before resize shapes', val_x.shape, val_y.shape)
    val_x = np.resize(val_x,(val_y.shape[0],val_x.shape[-1]))
    #val_x = np.concatenate((val_x,transformedData2),axis=0)
    print('Before plotting shapes', val_x.shape, val_y.shape)
    #val_x = val_x[actualClassData]
    #val_y = val_y[actualClassData]
    preparedVal_x, preparedVal_y = utils.prepareDataForPlotting(val_x,val_y,1)
    pcaReduced = utils.reduceDimensions(preparedVal_x,method='PCA',perplexity=30)
    fileName = 'smallTestField/validate_PCA.png'
    utils.plotHelper(fileName,pcaReduced,preparedVal_y)
    
    tsneReduced = utils.reduceDimensions(preparedVal_x,method='TSNE',perplexity=100)
    fileName = 'smallTestField/'+str(numberMelBins)+'Bin_'+str(hidden)+'hid_'+str(nfft)+'fft_'+'validate_TSNE.png'
    utils.plotHelper(fileName,tsneReduced,preparedVal_y)
    
    val_x = val_x[actualClassData]
    val_y = val_y[actualClassData]
    preparedVal_x2, preparedVal_y2 = utils.prepareDataForPlotting(val_x,val_y,1)
    tsneReduced2 = utils.reduceDimensions(preparedVal_x2,method='TSNE',perplexity=30)
    fileName2 = 'smallTestField/classOnly_'+str(numberMelBins)+'Bin_'+str(hidden)+'hid_'+str(nfft)+'fft_'+'validate_TSNE.png'
    utils.plotHelper(fileName2,tsneReduced2,preparedVal_y2)
    #import code
    #code.interact(local=locals())
