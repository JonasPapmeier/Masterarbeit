# -*- coding: utf-8 -*-
"""
Created on Wed May 31 13:46:20 2017

@author: 1papmeie
"""

import keras.backend as K
import numpy as np
from keras.models import Sequential,Model
from keras.layers.recurrent import LSTM, GRU
from keras.optimizers import SGD, Nadam
from keras.layers.core import Dense, Activation, Dropout,Lambda
from keras.layers import Input, ZeroPadding1D
from keras.layers.wrappers import TimeDistributed
from keras import regularizers
import os
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MiniBatchKMeans
import pickle
from GWR import readNetwork, GWR
import config_


def sumSquaredError(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true), axis=-1)

def createRNN_AE(num_layers,num_hidden,input_dim,output_dim,lr,sigma):
    model = Sequential()
    model.add(Dropout(sigma,input_shape=(None,int(input_dim[-1]))))
    #layerSizes = np.linspace(input_dim[-1]*1.5,num_hidden,num=num_layers,endpoint=False)
    layerSizes = [128 for i in range(num_layers)]
    for i in range(num_layers):
        model.add(GRU(int(layerSizes[i]),return_sequences=True,implementation=1))
    model.add(TimeDistributed(Dense(num_hidden,activation='linear')))
    model.add(Lambda(lambda x: x/K.sqrt(K.maximum(K.sum(K.square(x),axis=-1,keepdims=True),1e-12))))
    for i in range(num_layers):
        model.add(GRU(int(layerSizes[i]),return_sequences=True,implementation=1))
    model.add(TimeDistributed(Dense(input_dim[-1],activation='linear')))
    #opt = SGD(lr=lr,momentum=0.9,nesterov=True)
    opt = Nadam(clipnorm=2.)
    model.compile(loss='mae', optimizer=opt, metrics=['mse'])
    model.summary()
    return model

def createSeqToSeq(num_layer,num_hidden,input_dim,output_dim,lr,sigma):
    encoderInputs = Input(shape=(None,int(input_dim[-1])))
    encoderInputsDrop = Dropout(sigma,noise_shape=(None,1,int(input_dim[-1])))(encoderInputs)
    #Black magic to reverse encoder input sequence
    #Lambda layer wrapping necessary as to make keras believe we have actual proper layers and inputs
    encoderInputsReversed = Lambda(lambda x: x[:,::-1,:])(encoderInputsDrop)
    #layerSizes = np.linspace(np.prod(input_dim[1:])*1,num_hidden,num=num_layer,endpoint=True)
    layerSizes = np.linspace(128,num_hidden,num=num_layer,endpoint=True)
    #layerSizes = [128 for i in range(num_layer)]
    if num_layer == 1:
        layerSizes = [128]
    currentLayerOutput = encoderInputsDrop
    for i in range(num_layer):
        if i < num_layer-1:
            encoder = GRU(int(layerSizes[i]),return_state=False,implementation=1,stateful=False,return_sequences=True,kernel_regularizer=regularizers.l2(0))
            encoderOutputs = encoder(currentLayerOutput)
            currentLayerOutput = encoderOutputs
            currentLayerOutput = Dropout(0.,noise_shape=(None,1,None))(currentLayerOutput)
        else:
            encoder = GRU(int(layerSizes[i]),return_state=True,implementation=1,stateful=False,return_sequences=False,kernel_regularizer=regularizers.l2(0))
            encoderOutputs, stateH = encoder(currentLayerOutput)
    
    embedding = Dense(num_hidden)
    embeddingOutput = embedding(stateH)
    embeddingL2Norm = Lambda(lambda x: x/K.sqrt(K.maximum(K.sum(K.square(x),axis=-1,keepdims=True),1e-12)))
    embeddingL2NormOutput = embeddingL2Norm(embeddingOutput)
    
    #To create teacher forcing input for decoder
    #i.e. decoder input lags behind 1 step
    delayLayer = ZeroPadding1D((1,0))
    decoderInputs = delayLayer(encoderInputs)
    decoderInputs = Lambda(lambda x: x[:,:-1,:])(decoderInputs)
    decoderInputs = Lambda(lambda x: K.zeros_like(x))(decoderInputs)

    for i in reversed(range(num_layer)):
        if i == num_layer-1:
            decoder = GRU(int(layerSizes[i]),return_state=False,implementation=1,return_sequences=True,stateful=False)
            gruOutputs = decoder(decoderInputs,initial_state=embeddingL2NormOutput)
            currentLayerOutput = gruOutputs
            currentLayerOutput = Dropout(0.,noise_shape=(None,1,None))(currentLayerOutput)
        else:
            decoder = GRU(int(layerSizes[i]),return_state=False,implementation=1,return_sequences=True,stateful=False)
            gruOutputs = decoder(currentLayerOutput)
            currentLayerOutput = gruOutputs
            currentLayerOutput = Dropout(0.,noise_shape=(None,1,None))(currentLayerOutput)
    decoderLinear = Dense(input_dim[-1])
    decoderLinearDist = TimeDistributed(decoderLinear)
    decoderOutputs = decoderLinearDist(currentLayerOutput)
    model = Model(encoderInputs,decoderOutputs)
    #opt = SGD(lr=lr,momentum=0.9,nesterov=True,clipnorm=5.)
    opt = Nadam(clipnorm=2.)
    model.compile(loss='mae', optimizer=opt, metrics=['mse'])
    model.summary()
    return model
    
def createSeqToSinglePrediction(num_layer,num_hidden,input_dim,output_dim,lr,sigma,classType):
    inputs = Input(shape=(None,int(input_dim[-1])))
    currentLayerOuput = inputs
    for i in range(num_layer):
        if i < num_layer-1:
            currentLayer = GRU(num_hidden,implementation=2,return_sequences=True,stateful=False)
            currentLayerOuput = currentLayer(currentLayerOuput)
        else:
            currentLayer = GRU(num_hidden,implementation=2,return_sequences=False,stateful=False)
            currentLayerOuput = currentLayer(currentLayerOuput)
    if classType == 'categorical':
        outputActivation = 'softmax'
        targetLoss = 'categorical_crossentropy'
        targetMetrics = ['categorical_accuracy']
    else:
        outputActivation = 'sigmoid'
        targetLoss = 'binary_crossentropy'
        targetMetrics = ['binary_accuracy']
    outputs = Dense(output_dim[-1],activation=outputActivation)(currentLayerOuput)
    model = Model(inputs,outputs)
    model.summary()
    #opt = SGD(lr=lr,momentum=0.9,nesterov=True)
    opt = Nadam(clipnorm=2.)
    model.compile(loss=targetLoss,optimizer=opt,metrics=targetMetrics)
    return model
    
def createLSTM(num_layer,num_hidden,input_dim,output_dim,lr,sigma,classType):
    model = Sequential()
    model.add(Dropout(sigma,input_shape=(None,int(input_dim[-1]))))
    for i in range(num_layer):
        model.add(LSTM(num_hidden,return_sequences=True,implementation=2,input_shape=(None,input_dim)))
        model.add(Dropout(sigma))
    if classType == 'categorical':
        outputActivation = 'softmax'
        targetLoss = 'categorical_crossentropy'
        targetMetrics = ['categorical_accuracy']
    else:
        outputActivation = 'sigmoid'
        targetLoss = 'binary_crossentropy'
        targetMetrics = ['binary_accuracy']
    model.add(TimeDistributed(Dense(output_dim[-1],activation=outputActivation)))
    opt = SGD(lr=lr,momentum=0.9,nesterov=True)
    model.compile(loss=targetLoss, optimizer=opt, metrics=targetMetrics)
    model.summary()
    return model
    
def createFeedForward(num_layer,num_hidden,input_dim,output_dim,lr,sigma,classType):
    inputs = Input(shape=(int(input_dim[-1]),))
    drop1 = Dropout(sigma)(inputs)
    currentLayer = drop1
    for i in range(num_layer):
        currentLayer = Dense(num_hidden,activation='relu')(currentLayer)
        currentLayer = Dropout(sigma)(currentLayer)
    if classType == 'categorical':
        outputActivation = 'softmax'
        targetLoss = 'categorical_crossentropy'
        targetMetrics = ['categorical_accuracy']
    else:
        outputActivation = 'sigmoid'
        targetLoss = 'binary_crossentropy'
        targetMetrics = ['binary_accuracy']
    outputs = Dense(output_dim[-1],activation=outputActivation)(currentLayer)
    model = Model(inputs=inputs,outputs=outputs)
    model.summary()
    opt = SGD(lr=lr,momentum=0.9,nesterov=True)
    #opt = Nadam()
    model.compile(loss=targetLoss,optimizer=opt,metrics=targetMetrics)
    return model

def createFeedForward_AE(num_layer,num_hidden,input_dim,output_dim,lr,sigma):
    inputs = Input(shape=(int(input_dim[-1]),))
    drop1 = Dropout(sigma)(inputs)
    currentLayer = drop1
    layerSizes = [min(input_dim[-1]*3,1000) for i in range(num_layer)]
    for size in layerSizes:
        currentLayer = Dense(int(size),activation='relu')(currentLayer)
    embedding = Dense(num_hidden)(currentLayer)
    embeddingL2Norm = Lambda(lambda x: x/K.sqrt(K.maximum(K.sum(K.square(x),axis=-1,keepdims=True),1e-12)))
    embeddingL2NormOutput = embeddingL2Norm(embedding)
    currentLayer = embeddingL2NormOutput
    for size in reversed(layerSizes):
        currentLayer = Dense(int(size),activation='relu')(currentLayer)
    outputs = Dense(input_dim[-1])(currentLayer)
    model = Model(inputs=inputs,outputs=outputs)
    model.summary()
    #opt = SGD(lr=lr,momentum=0.9,nesterov=True)
    opt = Nadam(clipnorm=2)
    model.compile(loss='mae',optimizer=opt,metrics=['mse'])
    return model
    
def loadModelWeights(model,modelName,modelType):
    print(modelName)
    if os.path.isfile(modelName):
        print(modelType)
        if modelType in ['RNN_AE','LSTM','RNN_AE_NEW','FF','FF_AE','SEQ_AE', 'SEQ']:
            print('loading weights')
            model.load_weights(modelName)
        elif modelType in ['GMM','KMeans'] or modelType.endswith('KMeans'):
            print('loading weights')
            model = pickle.load(open(modelName,'rb'))
        elif modelType in ['GWR'] or modelType.endswith('GWR'):
            print('loading weights')
            model = readNetwork(modelName)
        return model
    return None

def loadModelKeras(modelName,modelType,num_layers=config_.layers,num_hidden=config_.hiddenNodes,input_dim=(1,50),output_dim=(1,11),
              lr=config_.lr,sigma=config_.sigma,classType=None):
    
    if modelType == 'RNN_AE':
        model = createRNN_AE(num_layers,num_hidden,input_dim,output_dim,lr,sigma)
    elif modelType == 'LSTM':
        model = createLSTM(num_layers,num_hidden,input_dim,output_dim,lr,sigma,classType)
    elif modelType == 'FF':
        model = createFeedForward(num_layers,num_hidden,input_dim,output_dim,lr,sigma,classType)
    elif modelType == 'FF_AE':
        model = createFeedForward_AE(num_layers,num_hidden,input_dim,output_dim,lr,sigma)
    elif modelType == 'SEQ_AE':
        model = createSeqToSeq(num_layers,num_hidden,input_dim,output_dim,lr,sigma)
    elif modelType == 'SEQ':
        model = createSeqToSinglePrediction(num_layers,num_hidden,input_dim,output_dim,lr,sigma,classType)
        
    savedModel = loadModelWeights(model,modelName,modelType)
    if savedModel:
        return savedModel
        
    return model
    
  
def loadModelSklearn(modelName,modelType,num_components=config_.numComponents,
                    tol=config_.GMM_tolerance,reg=config_.regularization,iters=config_.epochs,
                    num_cluster=config_.numCluster,batch_size=config_.batchSize,
                    patience=config_.patience,reassign=config_.reassignment):
    savedModel = loadModelWeights(None,modelName,modelType)
    if savedModel:
        return savedModel                        
    
    if modelType == 'GMM':
        model = GaussianMixture(n_components=num_components,max_iter=iters,
                    warm_start=True,verbose=1, verbose_interval=20,n_init=1,
                    covariance_type='diag',tol=tol,reg_covar=reg,init_params='kmeans')
    elif modelType.endswith('KMeans'):
        model = MiniBatchKMeans(n_clusters=num_cluster, init='k-means++', max_iter=iters, batch_size=batch_size,
                    verbose=1, compute_labels=True, random_state=1337, tol=tol,
                    max_no_improvement=patience, init_size=None, n_init=6, reassignment_ratio=reassign)
           
    return model
    
def loadModelGWR(modelName,modelType,trainData,epochs=config_.epochs,maxNodes=config_.maxNodes,maxNeighbours=config_.maxNeighbours,
                 maxAge=config_.maxAge,habThres=config_.habThres,insThres=config_.insThres,
                 epsB=config_.epsilonB,epsN=config_.epsilonN,tauB=config_.tauB,
                 tauN=config_.tauN):
    savedModel = loadModelWeights(None,modelName,modelType)
    if savedModel:
        return savedModel
        
    model = GWR(trainData,max_epochs=epochs,hab_threshold=habThres,insertion_threshold=insThres,
                epsilon_b=epsB,epsilon_n=epsN,tau_b=tauB,tau_n=tauN,MAX_NODES=maxNodes,
                MAX_NEIGHBOURS=maxNeighbours,MAX_AGE=maxAge)
    return model
