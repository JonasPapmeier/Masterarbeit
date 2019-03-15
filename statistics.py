# -*- coding: utf-8 -*-
"""
Created on Tue May 30 17:49:57 2017

@author: brummli
"""

import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from dataLoader import dataPrep

"""
Calculates standard statistics
:params:
    data: numpy array containing the data of form (examples,timesteps,dims)
:return:
    tuple containing (mean,var,min,max)
"""
def totalStats(data):
    mean = np.mean(data)
    var = np.var(data)
    minimum = np.min(data)
    maximum = np.max(data)
    return (mean,var,minimum,maximum)


"""
Calculates example wise standard statistics
:params:
    data: numpy array containing the data of form (examples,timesteps,dims)
:return:
    tuple containing (mean,var,min,max)
"""
def exampleStats(data):
    if len(data.shape) == 3:
        mean = np.mean(data,axis=(1,2))
        var = np.var(data,axis=(1,2))
        minimum = np.min(data,axis=(1,2))
        maximum = np.max(data,axis=(1,2))
    elif len(data.shape) == 2:
        mean = np.mean(data,axis=(1))
        var = np.var(data,axis=(1))
        minimum = np.min(data,axis=(1))
        maximum = np.max(data,axis=(1))
    else:
        #Make sure there are values to return so that plotting doesn't produce an error
        mean = 0
        var = 0
        minimum = 0
        maximum = 0
    return (mean,var,minimum,maximum)

"""
Calculates time wise standard statistics
:params:
    data: numpy array containing the data of form (examples,timesteps,dims)
:return:
    tuple containing (mean,var,min,max)
"""
def timeStats(data):
    if len(data.shape) == 3:
        mean = np.mean(data,axis=(0,2))
        var = np.var(data,axis=(0,2))
        minimum = np.min(data,axis=(0,2))
        maximum = np.max(data,axis=(0,2))
    else:
        #Make sure there are values to return so that plotting doesn't produce an error
        mean = 0
        var = 0
        minimum = 0
        maximum = 0
    return (mean,var,minimum,maximum)

"""
Calculates feature wise standard statistics
:params:
    data: numpy array containing the data of form (examples,timesteps,dims)
:return:
    tuple containing (mean,var,min,max)
"""
def featureStats(data):
    if len(data.shape) == 3:
        mean = np.mean(data,axis=(0,1))
        var = np.var(data,axis=(0,1))
        minimum = np.min(data,axis=(0,1))
        maximum = np.max(data,axis=(0,1))
    elif len(data.shape) == 2:
        mean = np.mean(data,axis=(0))
        var = np.var(data,axis=(0))
        minimum = np.min(data,axis=(0))
        maximum = np.max(data,axis=(0))
    else:
        #Make sure there are values to return so that plotting doesn't produce an error
        mean = 0
        var = 0
        minimum = 0
        maximum = 0
    return (mean,var,minimum,maximum)
    
def producePlot(func,data,descString,log=True):
    processed = func(data)
    fig = plt.figure()
    if log:
        plt.yscale('log')
    plt.plot(processed[0],'go-',label='mean')
    plt.plot(processed[1],'ro-',label='var')
    plt.plot(processed[2],'bo-',label='min')
    plt.plot(processed[3],'ko-',label='max')
    plt.legend()
    plt.savefig('stats/'+descString+'_'+func.__name__+'.png')
    plt.close(fig)
    
def plotWrapper(data,descString,log=True):
    producePlot(totalStats,data,descString,log=log)
    producePlot(exampleStats,data,descString,log=log)
    producePlot(timeStats,data,descString,log=log)
    producePlot(featureStats,data,descString,log=log)

if __name__ == '__main__':
    loader = dataPrep()
    dev_data_x,dev_data_y = loader.prepareDevTestSet(loader.devData)
    train_data_x,train_data_y = loader.prepareTrainSet(loader.augment(loader.trainData,0.8,1.2,3,0.8,1.2,3))
    
    fig = plt.figure(1)
    plt.boxplot(np.mean(np.mean(train_data_x,axis=1),axis=1))
    fig.show()
    
    fig = plt.figure(2)
    plt.boxplot(np.mean(train_data_x,axis=1).T)
    fig.show()
    
