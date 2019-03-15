# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 17:44:49 2018

@author: brummli
"""

import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TerminateOnNaN
from keras import backend as K
import time
from dataLoader import MARCHIdataPrep, DCASE2dataPrep, UrbanSound8KdataPrep
from modelLoader import loadModelKeras, loadModelSklearn, loadModelGWR
import audioProcessing
import config_
import sklearn.metrics as metrics
import sklearn.neighbors
from abc import ABC, abstractmethod
import pickle
import utils
import evaluation
from multiprocessing import Pool
from librosa.output import write_wav
import scipy.signal
import os


#Is class hierachy really a good way to implement this?
class Experiment(ABC):
    
    #Top init should only assign feature extraction, and train/valid/test params
    @abstractmethod
    def __init__(self,args):
        self.modelType = args.modelType 
        self.dataset = args.dataset
        
        self.preModelName = args.modelName

        self.epochs = args.epochs
        
        self.lengthScene = args.lengthScene
        self.lengthExample = args.lengthExample
        self.hopExample = args.hopExample
        self.eventsPerScene = args.numEvents
        self.method = args.method
        self.n_fft = args.n_fft
        self.hop = args.hop
        self.n_bin = args.n_bin
        self.augConfig = args.augmentation
        self.modelsFolder = args.modelsFolder
        self.logFolder = args.logFolder
        self.globalLog = args.globalLog
        self.globalCsv = args.globalCsv
        
        self.loader = None
        self.augmentedTrainData = None
        self.augmentedHoldoutData = None
        self.trainSet = None
        self.validSet = None
        self.testSet = None
        self.model = None
        self.changedModelName = False
        
        self.log_power = args.log_power
        self.deriv = args.derivate
        self.frameEnergy = args.frameEnergy
        self.predDelay = args.prediction_delay
        self.beta = args.beta
        self.max_dB = args.max_dB
        
        Experiment.actualizeModelName(self)
        
        
    def log(self,string):
        with open(self.globalLog,'a') as f:
            f.write(string)
        f.close()
        
    def logCsv(self,modelName,dataName,metrics):
        with open(self.globalCsv,'a') as f:
            fullData = list(metrics)
            fullData.insert(0,modelName)
            fullData.insert(1,dataName)
            fullData.append('\n')
            f.write(','.join(map(str,fullData)))
        f.close()
        
    #Subclasses have to extend super by own specific parameters
    @abstractmethod
    def actualizeModelName(self):
        self.modelName = (self.dataset+'_sceLen'+str(self.lengthScene)+'_sampLen'+str(self.lengthExample)+'_sampHop'+str(self.hopExample)+'_ev'+str(self.eventsPerScene)+
        '_'+self.method+'_fft'+str(self.n_fft)+'_hop'+str(self.hop)+'_bin'+str(self.n_bin)+
        '_aug'+str(self.augConfig)+'_lp'+str(self.log_power)+'_d'+str(self.deriv)+
        '_fe'+str(self.frameEnergy)+'_predD'+str(self.predDelay))
        self.changedModelName = True
        
    def loadDataStats(self,modelName):
        statsFileName = self.modelsFolder + 'ST_' + modelName+'.npz'
        if os.path.isfile(statsFileName):
            print('Loading stats',statsFileName)
            with np.load(statsFileName) as statsFile:
                preMean = statsFile['mean']
                preStd = statsFile['std']
                preMin = statsFile['minimum']
                preMax = statsFile['maximum']
                preMeanTarget = statsFile['meanTarget']
                preStdTarget = statsFile['stdTarget']
            return (preMean,preStd,preMin,preMax,preMeanTarget,preStdTarget)
        else:
            print('No stats file found')
            return None
        
    def saveDataStats(self,modelName,stats):
        statsFileName = self.modelsFolder + 'ST_' + modelName
        print('Saving stats',statsFileName)
        np.savez(statsFileName,mean=stats[0],std=stats[1],minimum=stats[2],maximum=stats[3],meanTarget=stats[4],stdTarget=stats[5])
        
    
    def setLoader(self):
        print('Create Loader')
        if self.dataset == 'Marchi':
            self.loader = MARCHIdataPrep(SR=config_.signalRate)
        elif self.dataset == 'DCASE2':
            self.loader = DCASE2dataPrep(SR=config_.signalRate,stats=self.stats)
        elif self.dataset == 'US8K':
            self.loader = UrbanSound8KdataPrep(SR=config_.signalRateUS8K,stats=self.stats)
        
    def forceLoader(self):
        if self.loader is None:
            print('Warning no data preloaded')
            self.loader = self.setLoader()
        
    def setAugData(self):
        minT, maxT, stepsT, minP, maxP, stepsP, minL, maxL, stepsL = config_.getAugmentConfig(self.augConfig)
        self.forceLoader()
        print('Augment data')
        self.augmentedTrainData = audioProcessing.augment(self.loader.trainData,self.loader.rate,minT,maxT,stepsT,minP,maxP,stepsP,minL,maxL,stepsL)
        
    def setAugConfig(self,augConfig):
        self.augConfig = augConfig
        self.actualizeModelName()
        self.setAugData()
    
    def createTrainSet(self,shuffle=True):
        self.forceLoader()
        print('Create training set')
        if self.augmentedTrainData is None:
            'Warning: No data augmentation was used'
            usedTrainData = self.loader.trainData
            usedHoldoutData = self.loader.holdoutData
        else:
            usedTrainData = self.augmentedTrainData
            usedHoldoutData = self.loader.holdoutData
        if self.modelType in ['GWR','KMeans','FF','FF_AE']:
            asFeatureVector = True
        else:
            asFeatureVector = False
        if self.modelType in ['SEQ_AE','SEQ']:
            temporalCompression = True
        else:
            temporalCompression = False
        #Placing the holdout fully in loader is probably better
        hold_x, hold_y, hold_y_ae = self.loader.prepareTrainSet(usedHoldoutData,lengthScene=self.lengthScene,
                                                      lengthExample=self.lengthExample,hopExample=self.hopExample,
                                                      eventsPerScene=self.eventsPerScene,asFeatureVector=asFeatureVector,temporalCompression=temporalCompression,
                                                      N_FFT=self.n_fft,hop=self.hop,log_power=self.log_power,deriv=self.deriv,
                                                      frameEnergy=self.frameEnergy,n_mels=self.n_bin,predDelay=self.predDelay,
                                                      shuffle=False,zNorm=False,minMaxNorm=False,l2Norm=False)
        self.trainSet = self.loader.prepareTrainSet(usedTrainData,lengthScene=self.lengthScene,lengthExample=self.lengthExample,
                                           hopExample=self.hopExample,eventsPerScene=self.eventsPerScene,
                                           asFeatureVector=asFeatureVector,temporalCompression=temporalCompression,N_FFT=self.n_fft,hop=self.hop,
                                           log_power=self.log_power,deriv=self.deriv,
                                           frameEnergy=self.frameEnergy,n_mels=self.n_bin,
                                           predDelay=self.predDelay,shuffle=shuffle,zNorm=self.zNorm,minMaxNorm=self.minMaxNorm,l2Norm=self.l2Norm)
        repeats = np.ones(len(self.loader.trainMean.shape),dtype=np.int32)
        repeats[-1] = hold_x.shape[-1]//self.loader.trainMean.shape[-1]
        if self.zNorm:
            hold_x = (hold_x - np.tile(self.loader.trainMean,repeats))/(np.tile(self.loader.trainStd,repeats)+1e-12)
            hold_y_ae = (hold_y_ae - np.tile(self.loader.trainMeanTarget,repeats))/(np.tile(self.loader.trainStdTarget,repeats)+1e-12)
        if self.minMaxNorm:
            hold_x = ((hold_x - np.tile(self.loader.trainMin,repeats)) / (np.tile((self.loader.trainMax - self.loader.trainMin),repeats)+1e-12))*2-1
            hold_y_ae = (hold_y_ae - np.tile(self.loader.trainMin,repeats)) / (np.tile((self.loader.trainMax - self.loader.trainMin),repeats)+1e-12)
        if self.l2Norm:
            hold_x = hold_x / np.sqrt(np.sum(hold_x**2,axis=-1,keepdims=True))
            hold_y_ae = hold_y_ae / np.sqrt(np.sum(hold_y_ae**2,axis=-1,keepdims=True))
        hold_x = np.squeeze(hold_x)
        hold_y = np.squeeze(hold_y)
        hold_y_ae = np.squeeze(hold_y_ae)
        self.holdoutSet = (hold_x,hold_y,hold_y_ae)
        
    def createValidSet(self):
        self.forceLoader()
        print('Create validation set')
        if self.modelType in ['GWR','KMeans','FF','FF_AE']:
            asFeatureVector = True
        else:
            asFeatureVector = False
        if self.modelType in ['SEQ_AE','SEQ']:
            temporalCompression = True
        else:
            temporalCompression = False
        self.validSet = self.loader.prepareValSet(N_FFT=self.n_fft,hop=self.hop,lengthScene=self.lengthScene,
                                                  lengthExample=self.lengthExample,
                                                  hopExample=self.hopExample,
                                                  asFeatureVector = asFeatureVector,temporalCompression=temporalCompression,
                                                  log_power=self.log_power,deriv=self.deriv,
                                                  frameEnergy=self.frameEnergy,n_mels=self.n_bin,
                                                  zNorm=self.zNorm,minMaxNorm=self.minMaxNorm,l2Norm=self.l2Norm)
                                                  
    def createTestSet(self):
        self.forceLoader()
        print('Create test set')
        if self.modelType in ['GWR','KMeans','FF','FF_AE']:
            asFeatureVector = True
        else:
            asFeatureVector = False
        if self.modelType in ['SEQ_AE','SEQ']:
            temporalCompression = True
        else:
            temporalCompression = False
        self.testSet = self.loader.prepareTestSet(N_FFT=self.n_fft,hop=self.hop,lengthScene=self.lengthScene,
                                                  lengthExample=self.lengthExample,
                                                  hopExample=self.hopExample,
                                                  asFeatureVector = asFeatureVector,temporalCompression=temporalCompression,
                                                  log_power=self.log_power,deriv=self.deriv,
                                                  frameEnergy=self.frameEnergy,n_mels=self.n_bin,
                                                  zNorm=self.zNorm,minMaxNorm=self.minMaxNorm,l2Norm=self.l2Norm)
    
    def plotTrainSet(self,dimReductionMethod='PCA',skips=1,perplexity=30,lr=200):
        if self.trainSet is None:
            self.createTrainSet()
        train_x, train_y, _ = self.trainSet
        preparedTrain_x,preparedTrain_y = utils.prepareDataForPlotting(train_x,train_y,skips)
        if preparedTrain_x.shape[-1] > 3:
            preparedTrain_x = utils.reduceDimensions(preparedTrain_x,method=dimReductionMethod,perplexity=perplexity)
        fileName = 'plots/datasets/train_'+self.dataset+'_'+str(dimReductionMethod)+'.png'
        utils.plotHelper(fileName,preparedTrain_x,preparedTrain_y)
    
    def plotValidSet(self,dimReductionMethod='PCA',skips=1,perplexity=30,lr=200):
        if self.validSet is None:
            self.createValidSet()
        val_x, val_y = self.validSet
        val_y = np.reshape(val_y,(-1,val_y.shape[-1]))
        val_y = val_y[self.hopExample//2::self.hopExample,:]
        print('Before plotting shapes', val_x.shape, val_y.shape)
        preparedVal_x, preparedVal_y = utils.prepareDataForPlotting(val_x,val_y,skips)
        if preparedVal_x.shape[-1] > 3:
            preparedVal_x = utils.reduceDimensions(preparedVal_x,method=dimReductionMethod,perplexity=perplexity)
        fileName = 'plots/datasets/validate_'+self.dataset+'_'+str(dimReductionMethod)+'.png'
        utils.plotHelper(fileName,preparedVal_x,preparedVal_y)
    
    def plotTestSet(self,dimReductionMethod='PCA',skips=1,perplexity=30):
        if self.testSet is None:
            self.createTestSet()
        test_x, test_y = self.testSet
        test_y = np.reshape(test_y,(-1,test_y.shape[-1]))
        test_y = test_y[self.hopExample//2::self.hopExample,:]
        print('Before plotting shapes', test_x.shape, test_y.shape)
        preparedTest_x, preparedTest_y = utils.prepareDataForPlotting(test_x,test_y,skips)
        if preparedTest_x.shape[-1] > 3:
            preparedTest_x = utils.reduceDimensions(preparedTest_x,method=dimReductionMethod,perplexity=perplexity)
        fileName = 'plots/datasets/test_'+self.dataset+'_'+str(dimReductionMethod)+'.png'
        utils.plotHelper(fileName,preparedTest_x,preparedTest_y)
        
    def plotTrainValidSet(self,dimReductionMethod='PCA',skips=1,perplexity=30,lr=200):
        if self.trainSet is None:
            self.createTrainSet()
        if self.validSet is None:
            self.createValidSet()
        val_y = self.validSet[1]
        val_y = np.reshape(val_y,(-1,val_y.shape[-1]))
        val_y = val_y[self.hopExample//2::self.hopExample,:]
        preparedTrain_x,preparedTrain_y = utils.prepareDataForPlotting(self.trainSet[0],self.trainSet[1],skips)
        preparedVal_x,preparedVal_y = utils.prepareDataForPlotting(self.validSet[0],val_y,skips)
        preparedVal_y = preparedVal_y + np.max(preparedTrain_y)
        combined_x = np.concatenate((preparedTrain_x,preparedVal_x),axis=0)
        combined_y = np.concatenate((preparedTrain_y,preparedVal_y),axis=0)
        randIndex = np.random.permutation(len(combined_x))
        combined_x = combined_x[randIndex]
        combined_y = combined_y[randIndex]
        if combined_x.shape[-1] > 3:
            combined_x = utils.reduceDimensions(combined_x,method=dimReductionMethod,perplexity=perplexity)
        fileName = 'plots/datasets/combined_'+self.dataset+'_'+str(dimReductionMethod)+'.png'
        utils.plotHelper(fileName,combined_x,combined_y)
        
    def plotTrainTestSet(self,dimReductionMethod='PCA',skips=1,perplexity=30):
        if self.trainSet is None:
            self.createTrainSet()
        if self.testSet is None:
            self.createTestSet()
        test_y = self.testSet[1]
        test_y = np.reshape(test_y,(-1,test_y.shape[-1]))
        test_y = test_y[self.hopExample//2::self.hopExample,:]
        preparedTrain_x,preparedTrain_y = utils.prepareDataForPlotting(self.trainSet[0],self.trainSet[1],skips)
        preparedTest_x,preparedTest_y = utils.prepareDataForPlotting(self.testSet[0],test_y,skips)
        preparedTest_y = preparedTest_y + np.max(preparedTrain_y)
        combined_x = np.concatenate((preparedTrain_x,preparedTest_x),axis=0)
        combined_y = np.concatenate((preparedTrain_y,preparedTest_y),axis=0)
        randIndex = np.random.permutation(len(combined_x))
        combined_x = combined_x[randIndex]
        combined_y = combined_y[randIndex]
        if combined_x.shape[-1] > 3:
            combined_x = utils.reduceDimensions(combined_x,method=dimReductionMethod,perplexity=perplexity)
        fileName = 'plots/datasets/combinedTrainTest_'+self.dataset+'_'+str(dimReductionMethod)+'.png'
        utils.plotHelper(fileName,combined_x,combined_y)
    
    def setDataParams(self,length,eventsPerScene,N_FFT,hop,n_bin):
        self.length=length
        self.eventsPerScene=eventsPerScene
        self.n_fft = N_FFT
        self.hop = hop
        self.n_bin
        self.actualizeModelName()
        self.setTrainData()
         
    @abstractmethod                                  
    def setModel(self):
        pass
    
    def setModelParams(self,modelType,layers,hiddenNodes,lr):
        self.modelType = modelType
        self.layers = layers
        self.hiddenNodes = hiddenNodes
        self.lr = lr
        self.actualizeModelName()
        self.setModel()
          
    @abstractmethod                     
    def train(self):
        print('Start Training')
        if self.trainSet is None:
            print('Forcing a train set')
            self.createTrainSet()
        if self.model is None or self.changedModelName:
            self.setModel()
        if not self.stats:
            #No stats for this model saved so far, create the file now
            print('Save stats for model')
            stats = (self.loader.trainMean,self.loader.trainStd,self.loader.trainMin,self.loader.trainMax,self.loader.trainMeanTarget,self.loader.trainStdTarget)
            self.saveDataStats(self.modelName,stats)
            self.stats=stats
      
    def validate(self,plot=False):
        if self.model is None:
            print('No model was selected. Aborting validation')
            return
        if self.validSet is None:
            self.createValidSet()
        print('Start Validation')
        if self.dataset == 'Marchi':
            self.validateMarchi(plot=plot)
        elif self.dataset == 'DCASE2':
            self.validateDCASE2(plot=plot)
        elif self.dataset == 'US8K':
            self.validateUrbanSound8K(plot=plot)
            
    def valHoldout(self,plot=False):
        if self.model is None:
            print('No model was selected. Aborting validation')
            return
        if self.holdoutSet is None:
            self.createTrainSet()
        print('Start Validation')
        if self.dataset == 'Marchi':
            self.valHoldoutMarchi(plot=plot)
        elif self.dataset == 'DCASE2':
            self.valHoldoutDCASE2(plot=plot)
        elif self.dataset == 'US8K':
            #UrbanSound has no need for extra holdout due to stratified folds
            self.validate(plot=plot)
    
    @abstractmethod
    def validateDCASE2(self,plot=False):
        pass
    
    @abstractmethod
    def validateUrbanSound8K(self,plot=False):
        pass
    
    @abstractmethod
    def valHoldoutDCASE2(self,plot=False):
        pass
        
    def postValidateMarchi(self,error,plot=False):
        valData_x, valData_y = self.validSet
        
        if plot:
            utils.plotAsLinePlot(error,'plots/val_error',999)
        median = np.median(error,axis=1,keepdims=True)
        beta = self.beta #Some value between 1 and 2? where does it come from? hyperparameter to tune rate of detection? how do they set this? validation set?
        binary_threshold = beta * median
        novelty_estimate = np.where(error>binary_threshold,1.,0.)
        if plot:
            utils.plotAsLinePlot(novelty_estimate,'plots/val_novelty',999)
        
        y_true = valData_y[:,self.predDelay:].flatten()
        y_pred = novelty_estimate.flatten()
        prec, rec, f1, sup = metrics.precision_recall_fscore_support(y_true,y_pred, average='binary')
        print('Precision', prec)
        print('Recall', rec)
        print('F-score', f1)
        print('Support', sup)
        hist = ('Precision '+str(prec)+' Recall '+str(rec)+' F-score '+str(f1)+' Support '+str(sup))
        logString = 'Validation values for '+self.modelName+'\n'+hist+'\n'
        self.log(logString)
        return (prec, rec, f1, sup)
        
    def postProcessingHelper(self,prediction):
        prediction = scipy.signal.medfilt(prediction,(1,7,1))
        return prediction

    def computeMetricsDCASE2(self,target,prediction):
        sampleMetrics = np.zeros((len(prediction),10))
        for num, data in enumerate(prediction):
            trueEventRoll = evaluation.convertRollTimeResolution(target[num],1.0/(self.loader.rate/self.hop),1.)
            predictedEventRoll = evaluation.convertRollTimeResolution(data,1.0/(self.loader.rate/self.hop),1.)
            sampleMetrics[num,:] = evaluation.segmentBasedMetrics(predictedEventRoll,trueEventRoll)
        m = np.sum(sampleMetrics,axis=0)
        aggregatedMetrics = evaluation.aggregation(m[0],m[1],m[2],m[3],m[4],m[5],m[6],m[7],m[8],m[9])
        hist = ('Precision '+str(aggregatedMetrics[0])+' Recall '+str(aggregatedMetrics[1])+
        ' F-score '+str(aggregatedMetrics[2])+' Error-rate '+str(aggregatedMetrics[3])+
        ' Substitutions '+str(aggregatedMetrics[4])+' Deletion '+str(aggregatedMetrics[5])+
        ' Insertions '+str(aggregatedMetrics[6]))
        return aggregatedMetrics, hist
    
    def postValidateDCASE2(self,prediction,plot=False):
        valData_x, valData_y = self.validSet
        prediction = self.postProcessingHelper(prediction)
        if plot:
            utils.plotAsPcolor(prediction,'plots/val_prediction',50)
            utils.plotAsPcolor(valData_y,'plots/val_target',50)
        metrics, hist = self.computeMetricsDCASE2(valData_y,prediction)
        logString = 'Validation values for '+self.modelName+'\n'+hist+'\n'
        self.log(logString)
        self.logCsv(self.modelName,'Validation DCASE2',metrics)
        print(hist)
        
    def postValidateDCASE2BySplit(self,prediction,maxLabel,labelOrder,plot=False):
        valData_x, valData_y = self.validSet
        prediction = self.postProcessingHelper(prediction)
        
        valData_y = valData_y[:,:,labelOrder]
        labelOrderWithNovel = labelOrder.copy()
        labelOrderWithNovel.append(max(labelOrder)+1)
        prediction = prediction[:,:,labelOrderWithNovel]
        print('Split',maxLabel)
        print('Reordering',valData_y.shape,prediction.shape)
        #Reorder valData_y and prediction right here
        
        #compress valData and prediction
        novelty = np.zeros((valData_y.shape[0],valData_y.shape[1],1))
        novelty[np.any(valData_y[...,maxLabel+1:],axis=-1,keepdims=True)] = 1
        predictNovelty = prediction[...,-1:]
        if plot:
            utils.plotAsPcolor(prediction,'plots/val_predictionWithNovelty',50)
            utils.plotAsPcolor(np.concatenate((valData_y[...,0:maxLabel+1],novelty),axis=-1),'plots/val_targetWithNovelty',50)
        classMetrics, classHistory = self.computeMetricsDCASE2(valData_y[...,0:maxLabel+1],prediction[...,0:maxLabel+1])
        noveltyMetrics, noveltyHistory = self.computeMetricsDCASE2(novelty,predictNovelty)
        
        scenesCutNovel_y = []
        scenesCutNovel_prediction = []
        for scene in range(valData_y.shape[0]):
            noveltyScene = novelty[scene]
            #Novel will be marked with one, so all zeros are normal background or normal class
            nonNovelParts = np.any(noveltyScene==0,axis=-1)
            nonNovel_y = valData_y[scene,nonNovelParts,0:maxLabel+1]
            scenesCutNovel_y.append(nonNovel_y)
            nonNovel_pred = prediction[scene,nonNovelParts,0:maxLabel+1]
            scenesCutNovel_prediction.append(nonNovel_pred)
        pureIncMetrics, pureIncHistory = self.computeMetricsDCASE2(scenesCutNovel_y,scenesCutNovel_prediction)
        
        classMetrics.extend(noveltyMetrics)
        classMetrics.extend(pureIncMetrics)
        joinedMetrics = classMetrics
        joinedHistory = classHistory+' Novelty '+noveltyHistory+' Inc '+pureIncHistory
        logString = 'Validation values for {} \n on split {} \n {} \n'.format(self.modelName,maxLabel,joinedHistory)
        self.log(logString)
        self.logCsv(self.modelName,'Validation DCASE2 split {}'.format(maxLabel), joinedMetrics)
        print(joinedHistory)
        
    def postValHoldoutDCASE2(self,prediction,plot=False):
        holdData_x, holdData_y, _ = self.holdoutSet
        if len(prediction.shape) < 3:
            prediction = prediction[np.newaxis,:,:]
            holdData_y = holdData_y[np.newaxis,:,:]
        prediction = self.postProcessingHelper(prediction)
        if plot:
            utils.plotAsPcolor(prediction,'plots/holdout_prediction',50)
            utils.plotAsPcolor(holdData_y,'plots/holdout_target',50)
        metrics, hist = self.computeMetricsDCASE2(holdData_y,prediction)
        logString = 'Holdout Validation values for '+self.modelName+'\n'+hist+'\n'
        self.log(logString)
        self.logCsv(self.modelName,'Holdout DCASE2',metrics)
        print(hist)
        
    def removeFakeLabelsUrbanSound8K(self,target,prediction):
        fakeLabelFlag = np.sum(target,axis=-1,keepdims=True) > 1
        predictionCleaned = np.where(fakeLabelFlag,np.zeros(prediction.shape[-1]),prediction)
        return predictionCleaned
        
    def joinExamplesUrbanSound8K(self,y):
        maxParts = int(np.ceil(4./self.lengthScene))
        y = np.reshape(y,(-1,maxParts,y.shape[-1]))
        yOut = np.mean(y,axis=1)
        return yOut
        
    def postValidateUrbanSound8K(self,prediction,plot=False):
        valData_x, valData_y = self.validSet
        if plot:
            #TODO: plotting functionality
            pass
        valData_y = self.removeFakeLabelsUrbanSound8K(valData_y,valData_y)
        valData_y = self.joinExamplesUrbanSound8K(valData_y)
        toOneHot = np.zeros_like(valData_y)
        toOneHot[np.arange(len(valData_y)),np.argmax(valData_y,axis=-1)] = 1
        valData_y = toOneHot
        print('Shape of evaluation data: {} and {}'.format(valData_y.shape,prediction.shape))
        prec, rec, f1, sup = metrics.precision_recall_fscore_support(valData_y,prediction, average='macro')
        acc = metrics.accuracy_score(valData_y,prediction)
        hist = ('Precision '+str(prec)+' Recall '+str(rec)+' F-score '+str(f1)+' Support '+str(sup)+' Accuracy '+str(acc))
        logString = 'Validation values for '+self.modelName+'\n'+hist+'\n'
        self.log(logString)
        self.logCsv(self.modelName,'Validation UrbanSound8K',[prec,rec,f1,sup,acc])
        print(hist)
        
    def test(self,plot=False):
        if self.model is None:
            print('No model was selected. Aborting test')
            return
        if self.testSet is None:
            print('Load test set')
            self.createTestSet()
        print('Start test')
        if self.dataset == 'Marchi':
            self.testMarchi(plot=plot)
        elif self.dataset == 'DCASE2':
            self.testDCASE2(plot=plot)
        elif self.dataset == 'US8K':
            self.testUrbanSound8K(plot=plot)
    
    @abstractmethod
    def testDCASE2 (self,plot=False):
        pass
    
    @abstractmethod
    def testUrbanSound8K(self,plot=False):
        pass
        
    def postTestDCASE2(self,prediction,plot=False):
        testData_x, testData_y = self.testSet
        prediction = self.postProcessingHelper(prediction)
        if plot:
            utils.plotAsPcolor(prediction,'plots/test_prediction',50)
            utils.plotAsPcolor(testData_y,'plots/test_target',50)
        metrics, hist = self.computeMetricsDCASE2(testData_y,prediction)
        print(hist)
        
    def postTestDCASE2BySplit(self,prediction,maxLabel,labelOrder,plot=False):
        testData_x, testData_y = self.testSet
        prediction = self.postProcessingHelper(prediction)
        
        #Reorder valData_y and prediction
        testData_y = testData_y[:,:,labelOrder]
        labelOrderWithNovel = labelOrder.copy()
        labelOrderWithNovel.append(max(labelOrder)+1)
        prediction = prediction[:,:,labelOrderWithNovel]
        print('Split',maxLabel)
        print('Reordering',testData_y.shape,prediction.shape)
        
        #compress valData and prediction
        novelty = np.zeros((testData_y.shape[0],testData_y.shape[1],1))
        novelty[np.any(testData_y[...,maxLabel+1:],axis=-1,keepdims=True)] = 1
        predictNovelty = prediction[...,-1:]
        if plot:
            utils.plotAsPcolor(prediction,'plots/test_predictionWithNovelty',50)
            utils.plotAsPcolor(np.concatenate((testData_y[...,0:maxLabel+1],novelty),axis=-1),'plots/test_targetWithNovelty',50)
        classMetrics, classHistory = self.computeMetricsDCASE2(testData_y[...,0:maxLabel+1],prediction[...,0:maxLabel+1])
        noveltyMetrics, noveltyHistory = self.computeMetricsDCASE2(novelty,predictNovelty)
        
        scenesCutNovel_y = []
        scenesCutNovel_prediction = []
        for scene in range(testData_y.shape[0]):
            noveltyScene = novelty[scene]
            #Novel will be marked with one, so all zeros are normal background or normal class
            nonNovelParts = np.any(noveltyScene==0,axis=-1)
            nonNovel_y = testData_y[scene,nonNovelParts,0:maxLabel+1]
            scenesCutNovel_y.append(nonNovel_y)
            nonNovel_pred = prediction[scene,nonNovelParts,0:maxLabel+1]
            scenesCutNovel_prediction.append(nonNovel_pred)
        pureIncMetrics, pureIncHistory = self.computeMetricsDCASE2(scenesCutNovel_y,scenesCutNovel_prediction)
        
        classMetrics.extend(noveltyMetrics)
        classMetrics.extend(pureIncMetrics)
        joinedMetrics = classMetrics
        joinedHistory = classHistory+' Novelty '+noveltyHistory+' Inc '+pureIncHistory
        logString = 'Test Values for {} \n on split {} \n {} \n'.format(self.modelName,maxLabel,joinedHistory)
        self.log(logString)
        self.logCsv(self.modelName,'Test DCASE2 split {}'.format(maxLabel),joinedMetrics)
        print(joinedHistory)
        
    def postTestUrbanSound8K(self,prediction,plot=False):
        testData_x, testData_y = self.testSet
        if plot:
            #TODO: plotting functionality
            pass
        testData_y = self.removeFakeLabelsUrbanSound8K(testData_y,testData_y)
        testData_y = self.joinExamplesUrbanSound8K(testData_y)
        toOneHot = np.zeros_like(testData_y)
        toOneHot[np.arange(len(testData_y)),np.argmax(testData_y,axis=-1)] = 1
        testData_y = toOneHot
        print('Shape of evaluation data: {} and {}'.format(testData_y.shape,prediction.shape))
        prec, rec, f1, sup = metrics.precision_recall_fscore_support(testData_y,prediction, average='macro')
        acc = metrics.accuracy_score(testData_y,prediction)
        hist = ('Precision '+str(prec)+' Recall '+str(rec)+' F-score '+str(f1)+' Support '+str(sup)+' Accuracy '+str(acc))
        print(hist)
    
class NNExperiment(Experiment):
    def __init__(self,args):
        super().__init__(args)
        self.layers= args.layers
        self.hiddenNodes = args.hiddenNodes
        self.lr = args.learningRate
        self.batch_size = args.batch_size
        self.sigma = args.sigma
        
        self.zNorm = True
        self.minMaxNorm = False
        self.l2Norm = False
        
        NNExperiment.actualizeModelName(self)
        self.stats = self.loadDataStats(self.modelName)
        
    
    def actualizeModelName(self):
        if self.preModelName:
            self.modelName = self.preModelName.replace('models/','')
        else:
            super().actualizeModelName()
            self.modelName = (self.modelType+'_lay'+str(self.layers)+'_nod'+str(self.hiddenNodes)+
            '_lr'+str(self.lr)+'_s'+str(self.sigma)+self.modelName)
                                       
    def setModel(self):
        print('load Model')
        if self.dataset in ['DCASE2']:
            classType = 'binary'
        elif self.dataset in ['US8K']:
            classType = 'categorical'
        else:
            classType = 'binary'
        if self.trainSet:
            bins = self.trainSet[0].shape
            outs = self.trainSet[1].shape
            self.model = loadModelKeras(self.modelsFolder+self.modelName,self.modelType,
                               num_layers=self.layers,num_hidden=self.hiddenNodes,
                               input_dim=bins,output_dim=outs,lr=self.lr,sigma=self.sigma,classType=classType)
        elif self.validSet:
            bins = self.validSet[0].shape
            outs = self.validSet[1].shape
            self.model = loadModelKeras(self.modelsFolder+self.modelName,self.modelType,
                               num_layers=self.layers,num_hidden=self.hiddenNodes,
                               input_dim=bins,output_dim=outs,lr=self.lr,sigma=self.sigma,classType=classType)
        elif self.testSet:
            bins = self.testSet[0].shape
            outs = self.testSet[1].shape
            self.model = loadModelKeras(self.modelsFolder+self.modelName,self.modelType,
                               num_layers=self.layers,num_hidden=self.hiddenNodes,
                               input_dim=bins,output_dim=outs,lr=self.lr,sigma=self.sigma,classType=classType)
        else:
            print("Warning: Couldn't interfere feature dims from data set, assume feature dims to be number of bins")
            self.model = loadModelKeras(self.modelsFolder+self.modelName,self.modelType,
                               num_layers=self.layers,num_hidden=self.hiddenNodes,
                               input_dim=(1,self.n_bin),output_dim=outs,lr=self.lr,sigma=self.sigma,classType=classType)
        self.changedModelName = False
        
    def trainAE(self,train_x,train_y,sysTime,callbacks):
        hold_x, _, hold_y_ae = self.holdoutSet
        if self.predDelay > 0:
            train_y = train_y[:,self.predDelay:,:]
            train_x = train_x[:,:-self.predDelay,:]
            hold_y = hold_y_ae[:,self.predDelay:,:]
            hold_x = hold_x[:,:-self.predDelay,:]
        print(train_x.shape,train_y.shape)
        history = self.model.fit(train_x,train_y,validation_data=(hold_x,hold_y_ae),verbose=2,batch_size=self.batch_size,epochs=self.epochs,callbacks=callbacks,shuffle=True)
        print('Finished Training')
        hist = ('val_mean_sqr_error: '+str(history.history['mean_squared_error'][-1])+' val_loss: '+
            str(history.history['loss'][-1]))
        logString = 'Finished '+self.modelName+'\nfrom '+sysTime+' with:\n'+hist+'\n'
        self.log(logString)
        
    def trainAEStream(self,train_x,train_y,sysTime,callbacks,incremental=False):
        hold_x, hold_y, hold_y_ae = self.holdoutSet
        if self.predDelay > 0:
            train_y = train_x[:,self.predDelay:,:]
            train_x = train_x[:,:-self.predDelay,:]
            hold_y = hold_y[:,self.predDelay:,:]
            hold_x = hold_x[:,:-self.predDelay,:]
        generator = self.loader.genTrainSetBatch(self.augmentedTrainData,batchSize=self.batch_size,lengthScene=self.lengthScene,lengthExample=self.lengthExample,
                                                 hopExample=self.hopExample,asFeatureVector=False,eventsPerScene=self.eventsPerScene,
                                                 N_FFT=self.n_fft,hop=self.hop,randSeed=1337,log_power=self.log_power,deriv=self.deriv,
                                                 frameEnergy=self.frameEnergy,n_mels=self.n_bin,predDelay=self.predDelay,autoencoder=True,
                                                 zNorm=self.zNorm,minMaxNorm=self.minMaxNorm)
        history = self.model.fit_generator(generator,steps_per_epoch=4000,epochs=self.epochs,verbose=2,callbacks=callbacks,validation_data=(hold_x,hold_y_ae),max_queue_size=10,workers=0)
        print('Finished Training')
        hist = ('val_mean_abs_error: '+str(history.history['mean_absolute_error'][-1])+' val_loss: '+
            str(history.history['loss'][-1]))
        logString = 'Finished '+self.modelName+'\nfrom '+sysTime+' with:\n'+hist+'\n'
        self.log(logString)
        
    def trainClassifier(self,train_x,train_y,sysTime,callbacks):
        history = self.model.fit(train_x,train_y,validation_data=(self.holdoutSet[0],self.holdoutSet[1]),verbose=2,batch_size=self.batch_size,
                                 epochs=self.epochs,callbacks=callbacks,shuffle=True)
        print('Finished Training')
        if self.dataset in ['US8K']:
            targetMetric = 'categorical_accuracy'
        elif self.dataset in ['DCASE2']:
            targetMetric = 'binary_accuracy'
        hist = ('val_accuracy: '+str(history.history['val_'+targetMetric][-1])+
            ' val_loss: '+str(history.history['val_loss'][-1])+' loss: '+
            str(history.history['loss'][-1])+' accuracy: '+str(history.history[targetMetric][-1]))
        logString = 'Finished '+self.modelName+'\nfrom '+sysTime+' with:\n'+hist+'\n'
        self.log(logString)
        
    def trainClassifierStream(self,train_x,train_y,sysTime,callbacks):
        generator = self.loader.genTrainSetBatch(self.augmentedTrainData,batchSize=self.batch_size,lengthScene=self.lengthScene,lengthExample=self.lengthExample,
                                                 hopExample=self.hopExample,asFeatureVector=True,temporalCompression=False,eventsPerScene=self.eventsPerScene,
                                                 N_FFT=self.n_fft,hop=self.hop,randSeed=1337,log_power=self.log_power,deriv=self.deriv,
                                                 frameEnergy=self.frameEnergy,n_mels=self.n_bin,predDelay=self.predDelay,autoencoder=False,
                                                 zNorm=self.zNorm,minMaxNorm=self.minMaxNorm)
        history = self.model.fit_generator(generator,steps_per_epoch=4000,epochs=self.epochs,verbose=2,callbacks=callbacks,validation_data=(self.holdoutSet[0],self.holdoutSet[1]),max_queue_size=30,workers=0)
        print('Finished Training')
        if self.dataset in ['US8K']:
            targetMetric = 'categorical_accuracy'
        elif self.dataset in ['DCASE2']:
            targetMetric = 'binary_accuracy'
        hist = ('val_accuracy: '+str(history.history['val_'+targetMetric][-1])+
            ' val_loss: '+str(history.history['val_loss'][-1])+' loss: '+
            str(history.history['loss'][-1])+' accuracy: '+str(history.history[targetMetric][-1]))
        logString = 'Finished '+self.modelName+'\nfrom '+sysTime+' with:\n'+hist+'\n'
        self.log(logString)
                             
    def train(self,incremental=False):
        super().train()
        
        train_x, train_y, train_y_ae = self.trainSet
        sysTime = time.asctime()
        logString = sysTime + ' starting:\n'+ self.modelName+' for '+str(self.epochs)+' epochs\n'
        self.log(logString)
        saveCallback = ModelCheckpoint(self.modelsFolder+self.modelName,monitor='val_loss',save_best_only=True,save_weights_only=False)
        stopCallback = EarlyStopping(monitor='loss',min_delta=0,patience=30)
        nanCallback = TerminateOnNaN()
        logCallback = CSVLogger(self.logFolder+self.modelName+'.csv',append=True)
        if '_AE' in self.modelType:
            if incremental:
                self.trainAEStream(train_x,train_y_ae,sysTime,[saveCallback,logCallback,nanCallback])
            else:
                self.trainAE(train_x,train_y_ae,sysTime,[saveCallback,logCallback,stopCallback,nanCallback])
        else:
            if incremental:
                self.trainClassifierStream(train_x,train_y,sysTime,[saveCallback,logCallback,nanCallback])
            else:
                self.trainClassifier(train_x,train_y,sysTime,[saveCallback,logCallback,stopCallback,nanCallback])
        #Basically the same as for AE and normal up to here
        #Question is then how to handle so far set values if we split into further sub functions
            
    def trainIncremental(self):
        self.train(incremental=True)
        
    def validateDCASE2(self,plot=False):
        super().validateDCASE2(plot=plot)
        valData_x, valData_y = self.validSet
        #Prediction changes with autoencoder, also normal validation not useful anymore, instead loss and metric of model on validation set
        if self.modelType in ['FF','FF_AE']:
            #valData_x = np.reshape(valData_x,(-1,valData_x.shape[-1]))
            pass
        if self.modelType.endswith('_AE') and not hasattr(self, 'combinedModelType'):
            valData_y_ae = valData_x
            if self.predDelay > 0:
                valData_y_ae = valData_y_ae[:,self.predDelay:,:]
                valData_x = valData_x[:,:-self.predDelay,:]
            metrics = self.model.evaluate(valData_x,valData_y_ae,batch_size=self.batch_size)
            hist = ('loss: '+str(metrics[0])+' mean_abs_error: '+
            str(metrics[1]))
            print(hist)
            logString = 'Validation values for '+self.modelName+'\n'+hist+'\n'
            self.log(logString)
            self.logCsv(self.modelName,'Validate DCASE2',metrics)
            return
        predicted_x = self.model.predict(valData_x, batch_size=self.batch_size)
        if self.modelType in ['FF','SEQ','FF_AE','SEQ_AE'] and self.hopExample > 1:
            predicted_x = np.repeat(predicted_x,self.hopExample,axis=0)
        print('Output shape is: {}'.format(predicted_x.shape))
        print('Target shape is: {}'.format(valData_y.shape))
        predicted_x = np.resize(predicted_x,valData_y.shape)
        predicted_x = np.where(predicted_x > self.beta, 1, 0)
        super().postValidateDCASE2(predicted_x,plot=plot)
        
    def validateUrbanSound8K(self,plot=False):
        valData_x, valData_y = self.validSet
        if self.modelType.endswith('_AE') and not hasattr(self, 'combinedModelType'):
            valData_y_ae = valData_x
            metrics = self.model.evaluate(valData_x,valData_y_ae,batch_size=self.batch_size)
            hist = ('loss: {}, mean_abs_error: {}'.format(metrics[0],metrics[1]))
            print(hist)
            logString = 'Validation values for '+self.modelName+'\n'+hist+'\n'
            self.log(logString)
            self.logCsv(self.modelName,'Validate UrbanSound8K',metrics)
            return
        predicted_y = self.model.predict(valData_x, batch_size=self.batch_size)
        print('Output shape is: {}'.format(predicted_y.shape))
        print('Target shape is: {}'.format(valData_y.shape))
        predicted_y = np.resize(predicted_y,valData_y.shape)
        predicted_y = self.removeFakeLabelsUrbanSound8K(valData_y,predicted_y)
        predicted_y = np.where(predicted_y > self.beta, 1, 0)
        #First decide on one values then average, otherwise beta thresholding is ill-defined for shorter examples
        predicted_y = self.joinExamplesUrbanSound8K(predicted_y)
        toOneHot = np.zeros_like(predicted_y)
        toOneHot[np.arange(len(predicted_y)),np.argmax(predicted_y,axis=-1)] = 1
        predicted_y = toOneHot
        super().postValidateUrbanSound8K(predicted_y,plot=plot)
        
    def valHoldoutDCASE2(self,plot=False):
        holdData_x, holdData_y, holdData_y_ae = self.holdoutSet
        #Prediction changes with autoencoder, also normal validation not useful anymore, instead loss and metric of model on validation set
        if self.modelType in ['FF','FF_AE']:
            #holdData_x = np.reshape(holdData_x,(-1,holdData_x.shape[-1]))
            pass
        if self.modelType.endswith('_AE') and not hasattr(self, 'combinedModelType'):
            if self.predDelay > 0:
                holdData_y = holdData_y_ae[:,self.predDelay:,:]
                holdData_x = holdData_x[:,:-self.predDelay,:]
            metrics = self.model.evaluate(holdData_x,holdData_y_ae,batch_size=self.batch_size)
            hist = ('loss: '+str(metrics[0])+' mean_abs_error: '+
            str(metrics[1]))
            print(hist)
            logString = 'Validation values for '+self.modelName+'\n'+hist+'\n'
            self.log(logString)
            self.logCsv(self.modelName,'Holdout DCASE2',metrics)
            return
        predicted_x = self.model.predict(holdData_x, batch_size=self.batch_size)
        print(predicted_x.shape)
        predicted_x = np.reshape(predicted_x,holdData_y.shape)
        predicted_x = np.where(predicted_x > self.beta, 1, 0)
        super().postValHoldoutDCASE2(predicted_x,plot=plot)
        
    def testDCASE2(self,plot=False):
        super().testDCASE2(plot=plot)
        testData_x, testData_y = self.testSet
        #Prediction changes with autoencoder, also normal validation not useful anymore, instead loss and metric of model on validation set
        if self.modelType.endswith('_AE') and not hasattr(self, 'combinedModelType'):
            testData_y_ae = testData_x
            if self.predDelay > 0:
                testData_y_ae = testData_y_ae[:,self.predDelay:,:]
                testData_x = testData_x[:,:-self.predDelay,:]
            metrics = self.model.evaluate(testData_x,testData_y_ae,batch_size=self.batch_size)
            hist = ('loss: '+str(metrics[0])+' mean_abs_error: '+
            str(metrics[1]))
            print(hist)
            logString = 'test values for '+self.modelName+'\n'+hist+'\n'
            self.log(logString)
            self.logCsv(self.modelName,'Test DCASE2',metrics)
            return
        predicted_x = self.model.predict(testData_x, batch_size=self.batch_size)
        if self.modelType in ['FF','SEQ','FF_AE','SEQ_AE'] and self.hopExample > 1:
            predicted_x = np.repeat(predicted_x,self.hopExample,axis=0)
        print('Output shape is: {}'.format(predicted_x.shape))
        print('Target shape is: {}'.format(testData_y.shape))
        predicted_x = np.resize(predicted_x,testData_y.shape)
        predicted_x = np.where(predicted_x > self.beta, 1, 0)
        super().postTestDCASE2(predicted_x,plot=plot)
        
    #TODO: refactor, redundancy with validateUrbanSound8K
    def testUrbanSound8K(self,plot=False):
        testData_x, testData_y = self.testSet
        if self.modelType.endswith('_AE') and not hasattr(self, 'combinedModelType'):
            testData_y_ae = testData_x
            metrics = self.model.evaluate(testData_x,testData_y_ae,batch_size=self.batch_size)
            hist = ('loss: {}, mean_abs_error: {}'.format(metrics[0],metrics[1]))
            print(hist)
            logString = 'Test values for '+self.modelName+'\n'+hist+'\n'
            self.log(logString)
            self.logCsv(self.modelName,'Test UrbanSound8K',metrics)
            return
        predicted_y = self.model.predict(testData_x, batch_size=self.batch_size)
        print('Output shape is: {}'.format(predicted_y.shape))
        print('Target shape is: {}'.format(testData_y.shape))
        predicted_y = np.resize(predicted_y,testData_y.shape)
        predicted_y = self.removeFakeLabelsUrbanSound8K(testData_y,predicted_y)
        predicted_y = np.where(predicted_y > self.beta, 1, 0)
        #First decide on one values then average, otherwise beta thresholding is ill-defined for shorter examples
        predicted_y = self.joinExamplesUrbanSound8K(predicted_y)
        toOneHot = np.zeros_like(predicted_y)
        toOneHot[np.arange(len(predicted_y)),np.argmax(predicted_y,axis=-1)] = 1
        predicted_y = toOneHot
        super().postTestUrbanSound8K(predicted_y,plot=plot)
        
    def removeOverlap(self,data,mode='skip'):
        if mode == 'skip':
            if self.lengthExample == self.hopExample:
                #No skips necessary
                skippedData = data
            elif self.hopExample == 1:
                skippedData = data[::self.lengthExample+1]
            else:
                print('Length hop combination not supported:', self.lengthExample, self.hopExample)
            transformedData = np.reshape(skippedData,(-1,data.shape[-1]//self.lengthExample))
        elif mode == 'middle':
            pass
        elif mode == 'average':
            pass
        else:
            pass
        transformedData = transformedData * self.loader.trainStd + self.loader.trainMean
        return transformedData
        
    def plotReconstDataHelper(self,data,nameAddition,produceWav=False,length=150,maxExample=10,numIter=0):
        if len(data.shape) == 2:
            data = data[:,:self.n_bin] #Cut off potential Frame Energy and Deriv because of scaling issues
            examples = min(data.shape[0]//length,maxExample)
        if len(data.shape) == 3:
            data = data[:,:,:self.n_bin]
            examples = min(np.prod(data.shape[0:2])//length,maxExample)
        data = np.resize(data,(examples,length,data.shape[-1]))
        utils.plotAsPcolor(data,'plots/DCASE2_reconst_'+nameAddition,maxExample)
        if produceWav:
            for example in range(examples):
                spec = audioProcessing.invertMelSpecTransformation(data[example],sr=self.loader.rate,
                                                               N_FFT=self.n_fft,hop=self.hop,log_power=self.log_power,
                                                               deriv=self.deriv,frameEnergy=self.frameEnergy,n_mels=self.n_bin)
                audio = audioProcessing.istftWithPhaseEstimation(spec,N_FFT=self.n_fft,HOP=self.hop,num_iteration=numIter,center=True)
                write_wav('plots/wav/DCASE2_reconst_'+nameAddition+str(example)+'.wav',audio,sr=self.loader.rate,norm=True)
    
    def plotAutoencoderReconst(self,mode='skip',produceWav=False,length=150,maxExample=10,numIter=0,withOriginal=False):
        #Call per hand everything else is annoying when making multiple calls
        train_x, _, _ = self.trainSet
        val_x, _= self.validSet
        print('Predict autoencoder output')
        predicted_train = self.model.predict(train_x,batch_size=self.batch_size)
        predicted_val = self.model.predict(val_x,batch_size=self.batch_size)
        if self.modelType in ['FF_AE'] and produceWav and self.lengthExample>1:
            predicted_train = self.removeOverlap(predicted_train,mode=mode)
            predicted_val = self.removeOverlap(predicted_val,mode=mode)
            orig_train = self.removeOverlap(train_x,mode=mode)
            orig_val = self.removeOverlap(val_x,mode=mode)
        else:
            orig_train = train_x * self.loader.trainStd + self.loader.trainMean
            orig_val = val_x * self.loader.trainStd + self.loader.trainMean
            predicted_train = predicted_train * self.loader.trainStd + self.loader.trainMean
            predicted_val = predicted_val * self.loader.trainStd + self.loader.trainMean
        print('Plot autoencoder output')
        self.plotReconstDataHelper(predicted_train,'train_hid'+str(self.hiddenNodes)+'_len'+str(self.lengthExample)+'_example',produceWav=produceWav,length=length,maxExample=maxExample,numIter=numIter)
        if withOriginal:
            self.plotReconstDataHelper(orig_train,'origTrain',produceWav=produceWav,length=length,maxExample=maxExample,numIter=numIter)
        self.plotReconstDataHelper(predicted_val,'val_hid'+str(self.hiddenNodes)+'_len'+str(self.lengthExample)+'_example',produceWav=produceWav,length=length,maxExample=maxExample,numIter=numIter)
        if withOriginal:
            self.plotReconstDataHelper(orig_val,'origVal',produceWav=produceWav,length=length,maxExample=maxExample,numIter=numIter)

            
class KMeansExperiment(Experiment):
    def __init__(self,args):
        super().__init__(args)
        self.numCluster = args.numCluster
        self.patience = args.patience
        self.tolerance = args.tolerance
        self.reassignment = args.reassignment
        self.batch_size = args.batch_size
        self.clusterLabelMapping = None
        
        self.modelName = (args.modelType+'_nCluster'+str(args.numCluster)+'_patience'+
                          str(args.patience)+'_tol'+str(args.tolerance)+'_reass'+
                          str(args.reassignment)+self.modelName)
        
    
    def actualizeModelName(self):
        if self.preModelName:
            self.modelName = self.preModelName
        else:
            super().actualizeModelName()
            self.modelName = (self.modelType+'_nCluster'+str(self.numCluster)+'_patience'+
                              str(self.patience)+'_tol'+str(self.tolerance)+'_reass'+
                              str(self.reassignment)+self.modelName)
                                       
    def setModel(self):
        print('load Model')
        self.model = loadModelSklearn(self.modelsFolder+self.modelName,self.modelType,
                               num_cluster=self.numCluster,tol=self.tolerance,
                               patience=self.patience,iters=self.epochs,
                               reassign=self.reassignment,batch_size=self.batch_size)
        self.changedModelName = False
       
    def setClusterLabelMapping(self):
        super().train()
        print('Compute cluster-label mapping')
        train_x, train_y, _ = self.trainSet
        cluster_assignments = self.model.labels_
        clusterLabels = []
        for clusterNum in range(self.numCluster):
            assignedDataIndex = np.where(cluster_assignments==clusterNum)
            sumLabel = np.sum(train_y[assignedDataIndex],axis=0)
            if np.max(sumLabel) > train_y[assignedDataIndex].shape[0]//2:
                clusterLabel = np.argmax(sumLabel)
            else:
                clusterLabel = -1
            clusterLabels.append(clusterLabel)
        self.clusterLabelMapping = clusterLabels
        
                             
    def train(self):
        super().train()
        
        train_x, train_y, _ = self.trainSet
        train_x = np.reshape(train_x,(train_x.shape[0],-1))
        sysTime = time.asctime()
        logString = sysTime + ' starting:\n'+ self.modelName+' for '+str(self.epochs)+' iterations\n'
        self.log(logString)
        self.model = self.model.fit(train_x)
        print('Finished Training')
        pickle.dump(self.model, open(self.modelsFolder+self.modelName, 'wb'))
        self.setClusterLabelMapping()
        logString = 'Finished '+self.modelName+'\nfrom '+sysTime+'\n'
        self.log(logString)
        
    def validateDCASE2(self,plot=False):
        super().validateDCASE2(plot=plot)
        if not self.clusterLabelMapping:
            self.setClusterLabelMapping()
        valData_x, valData_y = self.validSet
        valData_x = np.reshape(valData_x,(-1,valData_x.shape[2]))
        predicted_x = self.model.predict(valData_x)        
        binary_x = np.zeros((valData_x.shape[0],1,valData_y.shape[2]))
        for i in range(binary_x.shape[0]):
            label = self.clusterLabelMapping[predicted_x[i]]
            if label >= 0:
                binary_x[i,:,label] = 1
        binary_x = np.repeat(binary_x,self.hopExample,axis=0)
        binary_x = np.resize(binary_x,valData_y.shape)
        super().postValidateDCASE2(binary_x,plot=plot)
        
    def valHoldoutDCASE2(self,plot=False):
        if not self.clusterLabelMapping:
            self.setClusterLabelMapping()
        holdData_x, holdData_y, _ = self.holdoutSet
        holdData_x = np.reshape(holdData_x,(-1,holdData_x.shape[-1]))
        predicted_x = self.model.predict(holdData_x)        
        binary_x = np.zeros((holdData_x.shape[0],1,holdData_y.shape[-1]))
        for i in range(binary_x.shape[0]):
            label = self.clusterLabelMapping[predicted_x[i]]
            if label >= 0:
                binary_x[i,:,label] = 1
        binary_x = np.repeat(binary_x,self.hopExample,axis=0)
        binary_x = np.resize(binary_x,holdData_y.shape)
        super().postValHoldoutDCASE2(binary_x,plot=plot)
        
        
    def testDCASE2(self,plot=False):
        pass #TODO:
        
        
class AutoencoderKMeansExperiment(NNExperiment,KMeansExperiment):
    def __init__(self,args):
        super().__init__(args)
        self.NN_batch_size = args.batch_size
        self.NN_lengthExample = args.lengthExample
        self.KMeans_batch_size = args.KMeans_batch_size
        self.KMeans_lengthExample = 1
        self.combinedModelType = args.modelType+'-KMeans'
        self.extractionLayer = args.extractionLayer
        self.extractionFunction = None
        
        Experiment.actualizeModelName(self)
        self.NNModelName = (args.modelType+'_layer'+str(args.layers)+
        '_nodes'+str(args.hiddenNodes)+'_lr'+str(args.learningRate)+
        '_sig'+str(args.sigma)+self.modelName)
        self.NNModel = None
        
        self.actualizeModelName()
    
    def actualizeModelName(self):
        super().actualizeModelName()
        self.modelName = (self.combinedModelType+'km_bs'+str(self.KMeans_batch_size)+
        'km_le'+str(self.KMeans_lengthExample)+self.modelName)
        
    def setNNModel(self):
        print('load Neural Net Model')
        normalModelName = self.modelName
        self.modelName = self.NNModelName
        NNExperiment.setModel(self)
        self.NNModel = self.model
        self.modelName = normalModelName
                                       
    def setModel(self):
        print('load Model')
        self.model = loadModelSklearn(self.modelsFolder+self.modelName,self.combinedModelType,
                               num_cluster=self.numCluster,tol=self.tolerance,
                               patience=self.patience,iters=self.epochs,
                               reassign=self.reassignment,batch_size=self.KMeans_batch_size)
        self.changedModelName = False
        
    def transformToHiddenRepresentation(self,data):
        if not self.extractionFunction:
            input_tensor = self.NNModel.layers[0].input
            layer_output = self.NNModel.layers[self.extractionLayer].output
            self.extractionFunction = K.function([input_tensor,K.learning_phase()],[layer_output])
        transformedData = self.extractionFunction([data,0])[0]
        return transformedData
    
    def createTrainSet(self,shuffle=True):
        super().createTrainSet(shuffle=shuffle)
        if not self.NNModel:
            self.setNNModel()
        train_x, train_y, train_y_ae = self.trainSet
        transformed_train_x = self.transformToHiddenRepresentation(train_x)
        transformed_train_x = np.reshape(transformed_train_x, (-1,self.KMeans_lengthExample,transformed_train_x.shape[2]))
        train_y = np.reshape(train_y, (-1,self.KMeans_lengthExample,train_y.shape[2]))
        if shuffle:
            randGen = np.random.RandomState(1337)
            randIndex = np.arange(train_y.shape[0])
            randGen.shuffle(randIndex)
            transformed_train_x = transformed_train_x[randIndex,:,:]
            train_y = train_y[randIndex,:,:]
        print(np.mean(transformed_train_x), np.std(transformed_train_x), np.min(transformed_train_x), np.max(transformed_train_x))
        self.trainSet = (transformed_train_x, train_y, train_y_ae)
        holdout_x, holdout_y, holdout_y_ae = self.holdoutSet
        transformed_holdout_x = self.transformToHiddenRepresentation(holdout_x)
        transformed_holdout_x = np.reshape(transformed_holdout_x, (-1,self.KMeans_lengthExample,transformed_holdout_x.shape[2]))
        holdout_y = np.reshape(holdout_y, (-1,self.KMeans_lengthExample,holdout_y.shape[2]))
        self.holdoutSet = (transformed_holdout_x, holdout_y, holdout_y_ae)
        
    def createValidSet(self):
        super().createValidSet()
        if not self.NNModel:
            self.setNNModel()
        val_x, val_y = self.validSet
        transformed_val_x = self.transformToHiddenRepresentation(val_x)
        print(np.mean(transformed_val_x), np.std(transformed_val_x), np.min(transformed_val_x), np.max(transformed_val_x))
        self.validSet = (transformed_val_x, val_y)
        
    def createTestSet(self):
        super().createTestSet()
        if not self.NNModel:
            self.setNNModel()
        test_x, test_y = self.testSet
        transformed_test_x = self.transformToHiddenRepresentation(test_x)
        transformed_test_x = np.reshape(transformed_test_x, (-1,self.KMeans_lengthExample,transformed_test_x.shape[2]))
        test_y = np.reshape(test_y, (-1,self.KMeans_lengthExample,test_y.shape[2]))
        self.testSet = (transformed_test_x, test_y)
                             
    def train(self):
        KMeansExperiment.train(self)

        
    def validateDCASE2(self,plot=False):
        self.lengthExample = self.KMeans_lengthExample
        KMeansExperiment.validateDCASE2(self,plot=plot)
        self.lengthExample = self.NN_lengthExample
        
    def valHoldoutDCASE2(self,plot=False):
        self.lengthExample = self.KMeans_lengthExample
        KMeansExperiment.valHoldoutDCASE2(self,plot=plot)
        self.lengthExample = self.NN_lengthExample
        
    def testDCASE2(self,plot=False):
        pass #TODO:
        
class GWRExperiment(Experiment):
    def __init__(self,args):
        if args.lengthExample != 1:
            print("Warning, example length should currently be 1 for GWR but is", args.lengthExample)
        super().__init__(args)
        self.maxNodes = args.maxNodes
        self.maxNeighbours = args.maxNeighbours
        self.maxAge = args.maxAge
        self.habThres = args.habituationThreshold
        self.insThres = args.insertThreshold
        self.epsB = args.epsilonB
        self.epsN = args.epsilonN
        self.tauB = args.tauB
        self.tauN = args.tauN
        self.nodeLabelMapping = None
        self.maxNorm = None
        
        self.zNorm = False
        self.minMaxNorm = True
        self.l2Norm = True
        
        GWRExperiment.actualizeModelName(self)
        self.stats = self.loadDataStats(self.modelName)
        
    
    def actualizeModelName(self):
        if self.preModelName:
            self.modelName = self.preModelName.replace('models/','')
        else:
            super().actualizeModelName()
            self.modelName = (self.modelType+'_mNod'+str(self.maxNodes)+'_mAge'+str(self.maxAge)+
                              '_hThr'+str(self.habThres)+'_iThr'+str(self.insThres)+'_epsB'+
                              str(self.epsB)+'_epsN'+str(self.epsN)+'_tauB'+str(self.tauB)+
                              '_tauN'+str(self.tauN)+self.modelName)
                                       
    def setModel(self):
        print('load Model')
        if self.trainSet:
            train_x, _, _ = self.trainSet
            train_x = np.reshape(train_x,(train_x.shape[0],-1))
            self.model = loadModelGWR(self.modelsFolder+self.modelName,self.modelType,
                                      train_x,epochs=self.epochs,maxNodes=self.maxNodes,
                                      maxNeighbours=self.maxNeighbours,maxAge=self.maxAge,
                                      habThres=self.habThres,insThres=self.insThres,
                                      epsB=self.epsB,epsN=self.epsN,tauB=self.tauB,
                                      tauN=self.tauN)
        else:
            print("Warning: No train data loaded, initialize GWR with dummy data")
            dummyData = np.zeros((2,1))
            self.model = loadModelGWR(self.modelsFolder+self.modelName,self.modelType,
                                      dummyData,epochs=self.epochs,maxNodes=self.maxNodes,
                                      maxNeighbours=self.maxNeighbours,maxAge=self.maxAge,
                                      habThres=self.habThres,insThres=self.insThres,
                                      epsB=self.epsB,epsN=self.epsN,tauB=self.tauB,
                                      tauN=self.tauN)
        self.changedModelName = False
    
    def setNodeLabelMappingByDistance(self):
        super().train()
        print('Compute node-label mapping')
        train_x, train_y, _ = self.trainSet
        train_x = np.reshape(train_x,(train_x.shape[0],-1))
        train_y = np.reshape(train_y,(train_y.shape[0],-1))
        with Pool(processes=None) as pool:
            winningNodes = list(pool.map(self.model.findWinners,train_x))
        combinedSortedByDistance = sorted(zip(winningNodes,train_y),key=lambda pair: pair[0][2],reverse=True)
        #Initialize all node labels as None
        self.nodeLabelMapping = [None for i in range(len(self.model.nodes))]
        #Assign for each node the label of shortest distance
        #which is the first appearance of the node as best matching unit in the sorted list
        for zippedPair in combinedSortedByDistance:
            winningNode = zippedPair[0][0]
            if self.nodeLabelMapping[winningNode] is None:
                self.nodeLabelMapping[winningNode] = zippedPair[1]
                self.model.nodes[winningNode].label = np.argmax(np.concatenate((np.zeros(1),zippedPair[1]),axis=-1))
        #Make sure that every node gets a correct default label
        for num, label in enumerate(self.nodeLabelMapping):
            if label is None:
                self.nodeLabelMapping[num] = np.zeros((train_y.shape[-1]))
                self.model.nodes[num].label = 0
    
    def setNodeLabelMappingSimple(self):
        print('Cmopute node-label mapping')
        self.nodeLabelMapping = [0 for i in range(len(self.model.nodes))]
        for number, node in enumerate(self.model.nodes):
            label = np.argmax(np.concatenate((np.zeros(1),node.label),axis=-1),axis=-1)
            labelArray = np.zeros(len(self.loader.labels))
            if label > 0: #zero is background
                labelArray[label-1] = 1
            self.nodeLabelMapping[number] = labelArray
            
    def setNodeLabelMappingMaxFreq(self,threshold=0.7):
        print('Compute node-label mapping by highest occurence')
        self.nodeLabelMapping = [0 for i in range(len(self.model.nodes))]
        for number, node in enumerate(self.model.nodes):
            labelArray = np.copy(node.label)
            #tau_b is the parameter controlling the amount of habituation per activation
            #hab = node.habn
            #fireCount = (np.log(1./1.05 - (1.-hab)) - np.log(1./1.05)) * -np.log(1.05/self.model.tau_b)*2.
            #Initial planned to use habituation value, but the calculation is numerical unstable
            fireCount = node.fireCount
            labelArray = labelArray/fireCount
            threshold = threshold
            #Only those that were more than threshold percent active
            #Threshold basically describes how many "missing" labels are okay
            #Note that overlapping sounds have both classes in label vector
            #Thus a node representing an overlapping sound can have them both
            #close to 1
            labelArray = np.where(labelArray>=threshold,1,0)
            self.nodeLabelMapping[number] = labelArray
            
    def setNodeLabelMapping(self):
        self.setNodeLabelMappingMaxFreq()
        #self.setNodeLabelMappingSimple()
        #self.setNodeLabelMappingByDistance()
                             
    def train(self):
        super().train()
        
        train_x, train_y, _ = self.trainSet
        train_x = np.reshape(train_x,(train_x.shape[0],-1))
        sysTime = time.asctime()
        logString = sysTime + ' starting:\n'+ self.modelName+' for '+str(self.epochs)+' iterations\n'
        self.log(logString)
        self.model.train(data=train_x,labels=train_y,epochs=self.epochs)
        print('Finished Training')
        self.model.save(self.modelsFolder+self.modelName)
        self.setNodeLabelMapping()
        logString = 'Finished '+self.modelName+'\nfrom '+sysTime+'\n'
        self.log(logString)
    
    def trainIncremental(self):
        if self.augmentedTrainData is None:
            'Warning: No data augmentation was used'
            usedTrainData = self.loader.trainData
        else:
            usedTrainData = self.augmentedTrainData
        sysTime = time.asctime()
        logString = sysTime + ' starting:\n'+ self.modelName+' for '+str(self.epochs)+' iterations\n'
        self.log(logString)
        datasplits = self.loader.splitDataByLabel(usedTrainData,self.loader.labels)
        examplesPerEpoch = 1000
        exampleArray = np.zeros((examplesPerEpoch,self.trainSet[0].shape[-1]))
        targetArray = np.zeros((examplesPerEpoch,self.trainSet[1].shape[-1]))
        for datasplit in range(len(datasplits)):
            generator = self.loader.genTrainSetExample(datasplits[datasplit],lengthExample=self.lengthExample,hopExample=self.hopExample,
                                           asFeatureVector=True,deriv=self.deriv,frameEnergy=self.frameEnergy,
                                           n_mels=self.n_bin,randSeed=1337+datasplit*100,N_FFT=self.n_fft,hop=self.hop,
                                           zNorm=True,minMaxNorm=True)
            for epoch in range(self.epochs):
                for example in range(examplesPerEpoch):
                    exampleArray[example],targetArray[example],_ = next(generator)
                print('Datasplit {}: Train epoch {}'.format(datasplit,epoch))
                self.model.train(data=exampleArray,labels=targetArray,epochs=1)
            self.model.save(self.modelsFolder+self.modelName)
            self.setNodeLabelMapping()
            self.model.plotWithLabel('plots/gwr/datasplit'+str(datasplit)+'.png',dimReductionMethod='PCA')
            self.valHoldout()
            self.validate()
        logString = 'Finished '+self.modelName+'\nfrom '+sysTime+'\n'
        self.log(logString)
       
    def predict(self,dataX,dataY,noNovelDetection=False):
        binary_y = np.zeros((dataX.shape[0],1,dataY.shape[-1]+1))
        with Pool(processes=None) as pool:
            winningNodes = list(pool.map(self.model.findWinners,dataX))
        for index, nodes in enumerate(winningNodes):
            winningNode, _, dist = nodes
            dist = np.exp(dist) 
            label = self.nodeLabelMapping[winningNode]
            if (dist > self.beta and self.model.nodes[winningNode].habn < self.habThres) or noNovelDetection:
                binary_y[index,:,:-1] = label
            else: #Mark as novelty
                binary_y[index,:,-1] = 1
        return binary_y
       
    def validateDCASE2(self,plot=False):
        super().validateDCASE2(plot=plot)
        if self.nodeLabelMapping is None:
            self.setNodeLabelMapping()
        valData_x, valData_y = self.validSet
        valData_x = np.reshape(valData_x,(-1,valData_x.shape[-1]))
        binary_y = self.predict(valData_x,valData_y)
        binary_y = binary_y[:,:,:-1] #Cut out novelty prediction for normal mode
        binary_y = np.repeat(binary_y,self.hopExample,axis=0)
        binary_y.resize(valData_y.shape)
        super().postValidateDCASE2(binary_y,plot=plot)
        
    def validateDCASE2BySplit(self,maxLabel,labelOrder,plot=False):
        if self.nodeLabelMapping is None:
            self.setNodeLabelMapping()
        valData_x, valData_y = self.validSet
        valData_x = np.reshape(valData_x,(-1,valData_x.shape[-1]))
        binary_y = self.predict(valData_x,valData_y)
        binary_y = np.repeat(binary_y,self.hopExample,axis=0)
        binary_y.resize(valData_y.shape[0],valData_y.shape[1],valData_y.shape[2]+1)
        print(binary_y.shape)
        super().postValidateDCASE2BySplit(binary_y,maxLabel,labelOrder,plot=plot)
        print('Repeat without novelty detection')
        binary_y = self.predict(valData_x,valData_y,noNovelDetection=True)
        binary_y = np.repeat(binary_y,self.hopExample,axis=0)
        binary_y.resize(valData_y.shape[0],valData_y.shape[1],valData_y.shape[2]+1)
        super().postValidateDCASE2BySplit(binary_y,maxLabel,labelOrder,plot=plot)
        
        
    def valHoldoutDCASE2(self,plot=False):
        if self.nodeLabelMapping is None:
            self.setNodeLabelMapping()
        holdData_x, holdData_y, _ = self.holdoutSet
        holdData_x = np.reshape(holdData_x,(-1,holdData_x.shape[-1]))
        binary_y = self.predict(holdData_x,holdData_y)
        binary_y = binary_y[:,:,:-1]
        binary_y.resize(holdData_y.shape)
        super().postValHoldoutDCASE2(binary_y,plot=plot)
        
    def validateUrbanSound8K(self,plot=False):
        valData_x, valData_y = self.validSet
        predicted_y = self.predict(valData_x,valData_y)
        
        print('Output shape is: {}'.format(predicted_y.shape))
        print('Target shape is: {}'.format(valData_y.shape))
        predicted_y = np.resize(predicted_y,valData_y.shape)
        predicted_y = self.removeFakeLabelsUrbanSound8K(valData_y,predicted_y)
        predicted_y = self.joinExamplesUrbanSound8K(predicted_y)
        toOneHot = np.zeros_like(predicted_y)
        toOneHot[np.arange(len(predicted_y)),np.argmax(predicted_y,axis=-1)] = 1
        predicted_y = toOneHot
        super().postValidateUrbanSound8K(predicted_y,plot=plot)
        
    def testDCASE2(self,plot=False):
        if self.nodeLabelMapping is None:
            self.setNodeLabelMapping()
        testData_x, testData_y = self.testSet
        testData_x = np.reshape(testData_x,(-1,testData_x.shape[-1]))
        binary_y = self.predict(testData_x,testData_y)
        binary_y = binary_y[:,:,:-1]
        binary_y = np.repeat(binary_y,self.hopExample,axis=0)
        binary_y.resize(testData_y.shape)
        super().postTestDCASE2(binary_y,plot=plot)
        
    def testDCASE2BySplit(self,maxLabel,labelOrder,plot=False):
        if self.nodeLabelMapping is None:
            self.setNodeLabelMapping()
        testData_x, testData_y = self.testSet
        testData_x = np.reshape(testData_x,(-1,testData_x.shape[-1]))
        binary_y = self.predict(testData_x,testData_y)
        binary_y = np.repeat(binary_y,self.hopExample,axis=0)
        binary_y.resize(testData_y.shape[0],testData_y.shape[1],testData_y.shape[2]+1)
        super().postTestDCASE2BySplit(binary_y,maxLabel,labelOrder,plot=plot)
        print('Repeat without novelty detection')
        binary_y = self.predict(testData_x,testData_y,noNovelDetection=True)
        binary_y = np.repeat(binary_y,self.hopExample,axis=0)
        binary_y.resize(testData_y.shape[0],testData_y.shape[1],testData_y.shape[2]+1)
        super().postTestDCASE2BySplit(binary_y,maxLabel,labelOrder,plot=plot)
        
    def testUrbanSound8K(self,plot=False):
        pass #TODO:



class AutoencoderGWRExperiment(NNExperiment,GWRExperiment):
    def __init__(self,args):
        super().__init__(args)
        self.NN_batch_size = args.batch_size
        self.NN_lengthExample = args.lengthExample
        print(self.NN_lengthExample)
        self.NN_hopExample = args.hopExample
        self.GWR_lengthExample = args.GWR_lengthExample
        self.GWR_hopExample = args.GWR_hopExample
        self.NN_modelType = args.modelType
        self.combinedModelType = args.modelType+'-GWR'
        self.extractionLayer = args.extractionLayer
        self.extractionFunction = None
        
        self.zNorm = True
        self.minMaxNorm = False
        self.l2Norm = False
        
        Experiment.actualizeModelName(self)
        if args.GWR_NNModelName:
            self.NNModelName = args.GWR_NNModelName.replace('models/','')
        else:
            self.NNModelName = (args.modelType+'_lay'+str(args.layers)+
                                '_nod'+str(args.hiddenNodes)+'_lr'+str(args.learningRate)+
                                '_s'+str(args.sigma)+self.modelName)
        self.NNModel = None
        
        self.actualizeModelName()
        self.stats = self.loadDataStats(self.NNModelName)
        self.transformedStats = self.loadDataStats(self.modelName)
        
    
    def actualizeModelName(self):
        if self.preModelName:
            self.modelName = self.preModelName.replace('models/','')
        else:
            aeModelName = self.NNModelName.replace('RNN_AE_NEW','')
            aeModelName = aeModelName.replace('RNN_AE','')
            aeModelName = aeModelName.replace('FF_AE','')
            aeModelName = aeModelName.replace('SEQ_AE','')
            self.modelName = (self.combinedModelType+'gwr_le'+str(self.GWR_lengthExample)
            +aeModelName+'_aug'+str(self.augConfig)+'_iT'+str(self.insThres)+'_mAge'+str(self.maxAge)+'_lE'+str(self.lengthExample)+'_hE'+str(self.hopExample))
            self.lengthExample = self.GWR_lengthExample
        
    def setNNModel(self):
        print('load Neural Net Model')
        normalModelName = self.modelName
        self.modelName = self.NNModelName
        NNExperiment.setModel(self)
        self.NNModel = self.model
        self.modelName = normalModelName
        
    def setModel(self):
        print('load Model')
        if self.trainSet:
            train_x, _, _ = self.trainSet
            train_x = np.reshape(train_x,(train_x.shape[0],-1))
            self.model = loadModelGWR(self.modelsFolder+self.modelName,self.combinedModelType,
                                      train_x,epochs=self.epochs,maxNodes=self.maxNodes,
                                      maxNeighbours=self.maxNeighbours,maxAge=self.maxAge,
                                      habThres=self.habThres,insThres=self.insThres,
                                      epsB=self.epsB,epsN=self.epsN,tauB=self.tauB,
                                      tauN=self.tauN)
        else:
            print("Warning: No train set loaded, initialize GWR with dummy data")
            dummyData = np.zeros((2,1))
            self.model = loadModelGWR(self.modelsFolder+self.modelName,self.combinedModelType,
                                      dummyData,epochs=self.epochs,maxNodes=self.maxNodes,
                                      maxNeighbours=self.maxNeighbours,maxAge=self.maxAge,
                                      habThres=self.habThres,insThres=self.insThres,
                                      epsB=self.epsB,epsN=self.epsN,tauB=self.tauB,
                                      tauN=self.tauN)
        self.changedModelName = False
        
    def transformToHiddenRepresentation(self,data):
        if not self.extractionFunction:
            input_tensor = self.NNModel.layers[0].input
            layer_output = self.NNModel.layers[self.extractionLayer].output
            self.extractionFunction = K.function([input_tensor,K.learning_phase()],[layer_output])
        transformedData = self.extractionFunction([data,0])[0]
        return transformedData
        
    def rescaleToNegOneOne(self,data):
        try:
            minimum = self.transformedStats[2]
            maximum = self.transformedStats[3]
            minMaxDiff = maximum-minimum
            minMaxDiff = np.where(minMaxDiff==0,1e-12,minMaxDiff)
            data = ((data-minimum)/(minMaxDiff)) * 2 - 1
        except AttributeError:
            print('''Warning missing min,max values, load training set\n
            If you see this warning more than once you are probably stuck in an infinite loop''')
            self.createTrainSet()
            data = self.minMaxNorm(data)
        return data
    
    def createTrainSet(self,shuffle=True):
        self.lengthExample = self.NN_lengthExample
        if self.NN_modelType in ['RNN_AE']:
            self.hopExample = self.NN_lengthExample
        super().createTrainSet(shuffle=shuffle)
        self.lengthExample = self.GWR_lengthExample
        if not self.NNModel:
            self.setNNModel()
        train_x, train_y, train_y_ae = self.trainSet
        transformed_train_x = self.transformToHiddenRepresentation(train_x)
        if not self.transformedStats:
            #Mean and std not needed for GWR
            transformedMean = np.zeros(1)
            transformedStd = np.zeros(1)
            transformedMin = np.min(transformed_train_x,keepdims=False)
            transformedMax = np.max(transformed_train_x,keepdims=False)
            transformedMeanTarget = np.zeros(1)
            transformedStdTarget = np.zeros(1)
            transformedStats = (transformedMean,transformedStd,transformedMin,transformedMax,transformedMeanTarget,transformedStdTarget)
            self.saveDataStats(self.modelName,transformedStats)
            self.transformedStats = transformedStats
        transformed_train_x = self.rescaleToNegOneOne(transformed_train_x)
        transformed_train_x = transformed_train_x[::self.GWR_hopExample]
        if self.NN_modelType in ['RNN_AE']:
            newLength = -1
        else:
            newLength = transformed_train_x.shape[0]//self.GWR_lengthExample
        transformed_train_x = np.resize(transformed_train_x, (newLength,self.GWR_lengthExample*transformed_train_x.shape[-1]))
        train_y = train_y[::self.GWR_hopExample]
        train_y = np.resize(train_y, (newLength,self.GWR_lengthExample,train_y.shape[-1]))
        train_y = np.sum(train_y,axis=1)
        train_y = np.where(train_y>=self.GWR_lengthExample/2,1,0)
        if shuffle:
            randGen = np.random.RandomState(1337)
            randIndex = np.arange(train_y.shape[0])
            randGen.shuffle(randIndex)
            transformed_train_x = transformed_train_x[randIndex]
            train_y = train_y[randIndex]
        print('Mean,std,min,max',np.mean(transformed_train_x), np.std(transformed_train_x), np.min(transformed_train_x), np.max(transformed_train_x))
        self.trainSet = (transformed_train_x, train_y, train_y_ae)
        holdout_x, holdout_y, holdout_y_ae = self.holdoutSet
        transformed_holdout_x = self.transformToHiddenRepresentation(holdout_x)
        transformed_holdout_x = self.rescaleToNegOneOne(transformed_holdout_x)
        if self.NN_modelType in ['RNN_AE']:
            newLength = -1
        else:
            newLength = transformed_holdout_x.shape[0]//self.GWR_lengthExample
        transformed_holdout_x = np.resize(transformed_holdout_x, (newLength,self.GWR_lengthExample*transformed_holdout_x.shape[-1]))
        holdout_y = np.resize(holdout_y, (newLength,self.GWR_lengthExample,holdout_y.shape[-1]))
        holdout_y = np.sum(holdout_y,axis=1)
        holdout_y = np.where(holdout_y>=self.GWR_lengthExample/2,1,0)
        self.holdoutSet = (transformed_holdout_x, holdout_y, holdout_y_ae)
        
    def createValidSet(self):
        self.lengthExample = self.NN_lengthExample
        if self.NN_modelType in ['RNN_AE']:
            self.hopExample = self.NN_lengthExample
        super().createValidSet()
        self.lengthExample = self.GWR_lengthExample
        if not self.NNModel:
            self.setNNModel()
        val_x, val_y = self.validSet
        transformed_val_x = self.transformToHiddenRepresentation(val_x)
        transformed_val_x = self.rescaleToNegOneOne(transformed_val_x)
        transformed_val_x = transformed_val_x[::self.GWR_hopExample]
        if self.NN_modelType in ['RNN_AE']:
            newLength = -1
        else:
            newLength = transformed_val_x.shape[0]//self.GWR_lengthExample
        transformed_val_x = np.resize(transformed_val_x, (newLength,self.GWR_lengthExample*transformed_val_x.shape[-1]))
        print('Mean,std,min,max',np.mean(transformed_val_x), np.std(transformed_val_x), np.min(transformed_val_x), np.max(transformed_val_x))
        self.validSet = (transformed_val_x, val_y)
        
    def createTestSet(self):
        self.lengthExample = self.NN_lengthExample
        super().createTestSet()
        self.lengthExample = self.GWR_lengthExample
        if not self.NNModel:
            self.setNNModel()
        test_x, test_y = self.testSet
        transformed_test_x = self.transformToHiddenRepresentation(test_x)
        transformed_test_x = self.rescaleToNegOneOne(transformed_test_x)
        transformed_test_x = transformed_test_x[::self.GWR_hopExample]
        if self.NN_modelType in ['RNN_AE']:
             newLength = -1
        else:
             newLength = transformed_test_x.shape[0//self.GWR_lengthExample]
        transformed_test_x = np.resize(transformed_test_x, (newLength,self.GWR_lengthExample*transformed_test_x.shape[-1]))
        print('Test mean,std,min,max', np.mean(transformed_test_x),np.std(transformed_test_x),np.min(transformed_test_x),np.max(transformed_test_x))
        print(transformed_test_x.shape,test_y.shape)
        self.testSet = (transformed_test_x, test_y)
                             
    def train(self):
        GWRExperiment.train(self)
        
    def trainIncremental(self):
        if self.augmentedTrainData is None:
            'Warning: No data augmentation was used'
            usedTrainData = self.loader.trainData
        else:
            usedTrainData = self.augmentedTrainData
        sysTime = time.asctime()
        logString = sysTime + ' starting:\n'+ self.modelName+' incremental for '+str(self.epochs)+' iterations per split\n'
        self.log(logString)
        #Hard coded class order because I don't want to implement another fixed seed random generator
        #they were actually generated with numpy random except for the first one
        orders=[[1,10,2,3,8,4,0,9,5,6,7],
                [0,7,8,4,6,1,5,10,2,9,3],
                [1,10,3,7,6,9,0,4,2,5,8]]
        origDatasplits = self.loader.splitDataByLabel(usedTrainData,self.loader.labels)
        for var, order in enumerate(orders):
            self.modelName = self.modelName+'_var'+str(var)
            self.setModel()
            for datasplit in range(len(order)):
                #Also possible to directly iterate over order, then the logging will follow the order, problem would then be the maxLabel for the validation function
                self.augmentedTrainData = origDatasplits[order[datasplit]]
                self.createTrainSet()
                
                train_x, train_y, _ = self.trainSet
                self.model.train(data=train_x,labels=train_y,epochs=self.epochs)
                print('Finished Training')
                self.model.save(self.modelsFolder+'Split'+str(datasplit)+self.modelName)
                self.setNodeLabelMapping()
                self.model.plotWithLabel('plots/gwr/datasplit'+str(datasplit)+'_var'+str(var)+'.png',dimReductionMethod='PCA')
                self.validateDCASE2BySplit(datasplit,order,plot=False)
                self.testDCASE2BySplit(datasplit,order,plot=False)
        logString = 'Finished '+self.modelName+'\nfrom '+sysTime+'\n'
        self.log(logString)
        self.augmentedTrainData = usedTrainData
        
    def validateDCASE2(self,plot=False):
        self.lengthExample = self.GWR_lengthExample
        self.hopExample = 1
        GWRExperiment.validateDCASE2(self,plot=plot)
        self.lengthExample = self.NN_lengthExample
        
    def validateDCASE2BySplit(self,maxLabel,labelOrder,plot=False):
        self.lengthExample = self.GWR_lengthExample
        self.hopExample = 1
        GWRExperiment.validateDCASE2BySplit(self,maxLabel,labelOrder,plot=plot)
        self.lengthExample = self.NN_lengthExample
        
    def valHoldoutDCASE2(self,plot=False):
        self.lengthExample = self.GWR_lengthExample
        self.hopExample = 1
        GWRExperiment.valHoldoutDCASE2(self,plot=plot)
        self.lengthExample = self.NN_lengthExample
        
    def validateUrbanSound8K(self,plot=False):
        self.lengthExample = self.GWR_lengthExample
        GWRExperiment.validateUrbanSound8K(self,plot=plot)
        self.lengthExample = self.NN_lengthExample
        
        
    def testDCASE2(self,plot=False):
        self.lengthExample = self.GWR_lengthExample
        self.hopExample = 1
        GWRExperiment.testDCASE2(self,plot=plot)
        self.lengthExample = self.NN_lengthExample
        
    def testDCASE2BySplit(self,maxLabel,labelOrder,plot=False):
        self.lengthExample = self.GWR_lengthExample
        self.hopExample = 1
        GWRExperiment.testDCASE2BySplit(self,maxLabel,labelOrder,plot=plot)
        self.lengthExample = self.NN_lengthExample
        
    def testUrbanSound8K(self,plot=False):
        pass #TODO:

            
class AutoencoderSmallNNExperiment(NNExperiment):
    def __init__(self,args):
        super().__init__(args)
        self.AE_batch_size = args.batch_size
        self.AE_lengthExample = args.lengthExample
        self.AE_hopExample = args.hopExample
        self.NNpred_lengthExample = args.NNpred_lengthExample
        self.NNpred_hopExample = args.NNpred_hopExample
        self.AE_modelType = args.modelType
        self.combinedModelType = args.modelType+'-NN'
        self.extractionLayer = args.extractionLayer
        self.extractionFunction = None
        
        self.NNpred_layers = args.NNpred_layers
        self.NNpred_hiddenNodes = args.NNpred_hiddenNodes
        self.NNpred_lr = args.NNpred_lr
        self.NNpred_sigma = 0.
        
        self.model = None
        
        self.zNorm = True
        self.minMaxNorm = False
        self.l2Norm = False
        
        Experiment.actualizeModelName(self)
        if args.AE_ModelName:
            self.AEModelName = args.AE_ModelName.replace('models/','')
        else:
            self.AEModelName = (args.modelType+'_lay'+str(args.layers)+
                                '_nod'+str(args.hiddenNodes)+'_lr'+str(args.learningRate)+
                                '_s'+str(args.sigma)+self.modelName)
        self.AEModel = None
        
        self.actualizeModelName()
        self.stats = self.loadDataStats(self.AEModelName)
        self.transformedStats = self.loadDataStats(self.modelName)
        
    
    def actualizeModelName(self):
        if self.preModelName:
            self.modelName = self.preModelName.replace('models/','')
        else:
            aeModelName = self.AEModelName.replace('RNN_AE_NEW','')
            aeModelName = aeModelName.replace('RNN_AE','')
            aeModelName = aeModelName.replace('FF_AE','')
            aeModelName = aeModelName.replace('SEQ_AE','')
            self.modelName = (self.combinedModelType+'nn_le'+str(self.NNpred_lengthExample)
            +aeModelName+'_lE'+str(self.lengthExample)+'_hE'+str(self.hopExample))
            self.lengthExample = self.NNpred_lengthExample
        
    def setAEModel(self):
        print('load Neural Net Autoencoder Model')
        normalModelName = self.modelName
        self.modelName = self.AEModelName
        NNExperiment.setModel(self)
        self.AEModel = self.model
        self.model = None
        self.modelName = normalModelName
        
    def setModel(self):
        if self.model is None:
            print('load Model')
            if self.dataset in ['DCASE2']:
                classType = 'binary'
            elif self.dataset in ['US8K']:
                classType = 'categorical'
            else:
                classType = 'binary'
            if self.trainSet:
                train_x, _, _ = self.trainSet
                train_x = np.reshape(train_x,(train_x.shape[0],-1))
                outs = self.trainSet[1].shape
                self.model = loadModelKeras(self.modelsFolder+self.modelName,'FF',
                                   num_layers=self.NNpred_layers,num_hidden=self.NNpred_hiddenNodes,
                                   input_dim=train_x.shape,output_dim=outs,
                                   lr=self.NNpred_lr,sigma=self.NNpred_sigma,classType=classType)
            else:
                print("Warning: No dataset loaded, can not create small NN classifier")
            self.changedModelName = False
        
    def transformToHiddenRepresentation(self,data):
        if not self.extractionFunction:
            input_tensor = self.AEModel.layers[0].input
            layer_output = self.AEModel.layers[self.extractionLayer].output
            self.extractionFunction = K.function([input_tensor,K.learning_phase()],[layer_output])
        transformedData = self.extractionFunction([data,0])[0]
        return transformedData
        
    def rescaleToVarOne(self,data):
        try:
            var = self.transformedStats[1]
            var = np.where(var==0,1e-12,var)
            data = data/var
        except AttributeError:
            print('''Warning: No variance found, load train set\n
            If you see this message more than once you are probably stuck in an infinite loop''')
            self.createTrainSet()
            data = self.rescaleToVarOne(data)
        return data
    
    def createTrainSet(self,shuffle=True):
        self.lengthExample = self.AE_lengthExample
        if self.AE_modelType in ['RNN_AE']:
            self.hopExample = self.AE_lengthExample
        super().createTrainSet(shuffle=shuffle)
        self.lengthExample = self.NNpred_lengthExample
        if not self.AEModel:
            self.setAEModel()
        train_x, train_y, train_y_ae = self.trainSet
        print('train_x shape: {}, train_y shape {}'.format(train_x.shape,train_y.shape))
        transformed_train_x = self.transformToHiddenRepresentation(train_x)
        if not self.transformedStats:
            print('Compute stats')
            transformedMean = np.zeros(1)
            transformedStd = np.std(transformed_train_x)
            transformedMin = np.zeros(1)
            transformedMax = np.zeros(1)
            transformedMeanTarget = np.zeros(1)
            transformedStdTarget = np.zeros(1)
            transformedStats = (transformedMean,transformedStd,transformedMin,transformedMax,transformedMeanTarget,transformedStdTarget)
            self.saveDataStats(self.modelName,transformedStats)
            self.transformedStats = transformedStats
        transformed_train_x = self.rescaleToVarOne(transformed_train_x)
        transformed_train_x = transformed_train_x[::self.NNpred_hopExample]
        if self.AE_modelType in ['RNN_AE']:
            newLength = -1
        else:
            newLength = transformed_train_x.shape[0]//self.NNpred_lengthExample
        transformed_train_x = np.reshape(transformed_train_x, (newLength,self.NNpred_lengthExample*transformed_train_x.shape[-1]))
        train_y = train_y[::self.NNpred_hopExample]
        train_y = np.reshape(train_y, (newLength,self.NNpred_lengthExample,train_y.shape[-1]))
        train_y = np.sum(train_y,axis=1)
        train_y = np.where(train_y>=self.NNpred_lengthExample/2.,1,0)
        if shuffle:
            randGen = np.random.RandomState(1337)
            randIndex = np.arange(train_y.shape[0])
            randGen.shuffle(randIndex)
            transformed_train_x = transformed_train_x[randIndex]
            train_y = train_y[randIndex]
        print('Mean,std,min,max',np.mean(transformed_train_x), np.std(transformed_train_x), np.min(transformed_train_x), np.max(transformed_train_x))
        self.trainSet = (transformed_train_x, train_y, train_y_ae)
        holdout_x, holdout_y, holdout_y_ae = self.holdoutSet
        transformed_holdout_x = self.transformToHiddenRepresentation(holdout_x)
        transformed_holdout_x = self.rescaleToVarOne(transformed_holdout_x)
        if self.AE_modelType in ['RNN_AE']:
            newLength = -1
        else:
            newLength = transformed_holdout_x.shape[0]//self.NNpred_lengthExample
        transformed_holdout_x = np.reshape(transformed_holdout_x, (newLength,self.NNpred_lengthExample*transformed_holdout_x.shape[-1]))
        holdout_y = np.reshape(holdout_y, (newLength,self.NNpred_lengthExample,holdout_y.shape[-1]))
        holdout_y = np.sum(holdout_y,axis=1)
        holdout_y = np.where(holdout_y>=self.NNpred_lengthExample/2.,1,0)
        self.holdoutSet = (transformed_holdout_x, holdout_y, holdout_y_ae)
        
    def createValidSet(self):
        self.lengthExample = self.AE_lengthExample
        if self.AE_modelType in ['RNN_AE']:
            self.hopExample = self.AE_lengthExample
        super().createValidSet()
        self.lengthExample = self.NNpred_lengthExample
        if not self.AEModel:
            self.setAEModel()
        val_x, val_y = self.validSet
        transformed_val_x = self.transformToHiddenRepresentation(val_x)
        transformed_val_x = self.rescaleToVarOne(transformed_val_x)
        transformed_val_x = transformed_val_x[::self.NNpred_hopExample]
        if self.AE_modelType in ['RNN_AE']:
            newLength = -1
        else:
            newLength = transformed_val_x.shape[0]//self.NNpred_lengthExample
        transformed_val_x = np.reshape(transformed_val_x, (newLength,self.NNpred_lengthExample*transformed_val_x.shape[-1]))
        print('Mean,std,min,max',np.mean(transformed_val_x), np.std(transformed_val_x), np.min(transformed_val_x), np.max(transformed_val_x))
        print(transformed_val_x.shape,val_y.shape)
        self.validSet = (transformed_val_x, val_y)
        
    def createTestSet(self):
        self.lengthExample = self.AE_lengthExample
        if self.AE_modelType in ['RNN_AE']:
            self.hopExample = self.AE_lengthExample
        super().createTestSet()
        if not self.AEModel:
            self.setAEModel()
        test_x, test_y = self.testSet
        transformed_test_x = self.transformToHiddenRepresentation(test_x)
        transformed_test_x = self.rescaleToVarOne(transformed_test_x)
        transformed_test_x = transformed_test_x[::self.NNpred_hopExample]
        if self.AE_modelType in ['RNN_AE']:
            newLength = -1
        else:
            newLength = transformed_test_x.shape[0]//self.NNpred_lengthExample
        transformed_test_x = np.reshape(transformed_test_x, (newLength,self.NNpred_lengthExample*transformed_test_x.shape[-1]))
        print('Mean,std,min,max',np.mean(transformed_test_x), np.std(transformed_test_x), np.min(transformed_test_x), np.max(transformed_test_x))
        print(transformed_test_x.shape,test_y.shape)
        self.testSet = (transformed_test_x, test_y)
                             
    def train(self):
        train_x, train_y, train_y_ae = self.trainSet
        sysTime = time.asctime()
        logString = sysTime + ' starting:\n'+ self.modelName+' for '+str(self.epochs)+' epochs\n'
        self.log(logString)
        
        history = self.model.fit(train_x,train_y,validation_data=(self.holdoutSet[0],self.holdoutSet[1]),verbose=2,batch_size=self.batch_size,
                                 epochs=self.epochs,shuffle=True)
        print('Finished Training')
        if self.dataset in ['US8K']:
            targetMetric = 'categorical_accuracy'
        elif self.dataset in ['DCASE2']:
            targetMetric = 'binary_accuracy'
        hist = ('val_accuracy: '+str(history.history['val_'+targetMetric][-1])+
            ' val_loss: '+str(history.history['val_loss'][-1])+' loss: '+
            str(history.history['loss'][-1])+' accuracy: '+str(history.history[targetMetric][-1]))
        logString = 'Finished '+self.modelName+'\nfrom '+sysTime+' with:\n'+hist+'\n'
        self.log(logString)
        
    def trainIncremental(self):
        self.trainClassifierStream()
    
    def trainClassifierStream(self):
        sysTime = time.asctime()
        logString = sysTime + ' starting:\n'+ self.modelName+' for '+str(self.epochs)+' epochs\n'
        self.log(logString)
        def toHidFeatWrapper():
            gen = self.loader.genTrainSetBatch(self.augmentedTrainData,batchSize=self.batch_size,lengthScene=self.lengthScene,lengthExample=self.AE_lengthExample,
                                                 hopExample=self.AE_hopExample,asFeatureVector=False,temporalCompression=True,eventsPerScene=self.eventsPerScene,
                                                 N_FFT=self.n_fft,hop=self.hop,randSeed=1337,log_power=self.log_power,deriv=self.deriv,
                                                 frameEnergy=self.frameEnergy,n_mels=self.n_bin,predDelay=self.predDelay,autoencoder=False,
                                                 zNorm=self.zNorm,minMaxNorm=self.minMaxNorm)
            while True:
                batchX, batchY = next(gen)
                hiddenX = self.transformToHiddenRepresentation(batchX)
                hiddenX = self.rescaleToVarOne(hiddenX)
                yield (hiddenX, batchY)
        generator = toHidFeatWrapper()
        history = self.model.fit_generator(generator,steps_per_epoch=4000,epochs=self.epochs,verbose=2,validation_data=(self.holdoutSet[0],self.holdoutSet[1]),max_queue_size=30,workers=0)
        print('Finished Training')
        if self.dataset in ['US8K']:
            targetMetric = 'categorical_accuracy'
        elif self.dataset in ['DCASE2']:
            targetMetric = 'binary_accuracy'
        hist = ('val_accuracy: '+str(history.history['val_'+targetMetric][-1])+
            ' val_loss: '+str(history.history['val_loss'][-1])+' loss: '+
            str(history.history['loss'][-1])+' accuracy: '+str(history.history[targetMetric][-1]))
        logString = 'Finished '+self.modelName+'\nfrom '+sysTime+' with:\n'+hist+'\n'
        self.log(logString)
        
        
    def validateDCASE2(self,plot=False):
        self.lengthExample = self.NNpred_lengthExample
        self.hopExample = 1
        super().validateDCASE2(plot=plot)
        self.lengthExample = self.AE_lengthExample
        
    def valHoldoutDCASE2(self,plot=False):
        self.lengthExample = self.NNpred_lengthExample
        self.hopExample = 1
        super().valHoldoutDCASE2(plot=plot)
        self.lengthExample = self.AE_lengthExample
        self.validate(plot=plot)
        
    def validateUrbanSound8K(self,plot=False):
        self.lengthExample = self.NNpred_lengthExample
        super().validateUrbanSound8K(plot=plot)
        self.lengthExample = self.AE_lengthExample
        
        
    def testDCASE2(self,plot=False):
        self.lengthExample = self.NNpred_lengthExample
        self.hopExample = 1
        super().testDCASE2(plot=plot)
        self.lengthExample = self.AE_lengthExample
        
    def testUrbanSound8K(self,plot=False):
        self.lengthExample = self.NNpred_lengthExample
        super().testUrbanSound8K(plot=plot)
        self.lengthExample = self.AE_lengthExample
