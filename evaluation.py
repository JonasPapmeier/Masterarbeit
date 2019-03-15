# -*- coding: utf-8 -*-
"""
Created on Mon May 29 15:23:08 2017

@author: 1papmeie
"""

import numpy as np
import math

"""
Converts a list of events into a framewise annotation
:params:
eventList: list of events in form [(onset,offset,label)]
timeResolution: target time resolution for representation in seconds
totalLenght: (int) total number of frames in representation
numLabels: (int) total number of different event labels in whole dataset
:return:
Event Roll as Numpy array
"""
def eventListToRoll(eventList,timeResolution,totalLength,numLabels):
    y_roll = np.zeros((totalLength,numLabels))
    
    for onset, offset, label in eventList:
        ind_onset = math.floor(onset/timeResolution)
        ind_offset = min(math.ceil(offset/timeResolution)-1,totalLength)
        y_roll[ind_onset:ind_offset,label] = 1
        
    return y_roll
    
"""
Convert a framewise annotation to a list of events
:params:
eventRoll: (numpy array) framwise annotation
timeResolution: time Resolution of event Roll in seconds
labelList: list of labels in form [(index,'labelName')]
"""
def eventRollToList(eventRoll,timeResolution,labelList,minDur=0.0001):
    eventList=[]
    eventRoll = np.diff(np.concatenate((np.zeros((1,eventRoll.shape[1])),eventRoll,np.zeros((1,eventRoll.shape[1])))),axis=0)
    ind_events = np.transpose(np.nonzero(eventRoll))
    for i in range(0,ind_events.shape[0]-1,2):
        onset = ind_events[i,0]*timeResolution
        offset = ind_events[i+1,0]*timeResolution
        if((offset/100) - (onset/100) > minDur):
            eventList.append((onset,offset,labelList[ind_events[i,1]]))
    return eventList
   
def convertRollTimeResolution(eventRoll,currentRes,targetRes):
    if targetRes <= currentRes:
        factor = round(currentRes/targetRes)
        outputEventRoll = np.repeat(eventRoll,factor,axis=0)
    else:
        factor = currentRes/targetRes
        outputEventRoll = np.zeros((round(eventRoll.shape[0]*factor),eventRoll.shape[1]))
        for i in range(outputEventRoll.shape[0]):
            outputEventRoll[i,:] = np.max(eventRoll[math.floor(i*1/factor):math.floor((i+1)*1/factor)],axis=0)
    return outputEventRoll
    
def segmentBasedMetrics(predictedEventRoll,annotatedEventRoll):
    assert predictedEventRoll.shape == annotatedEventRoll.shape
    totalNtp = 0
    totalNtn = 0
    totalNfp = 0
    totalNfn = 0
    totalNref = 0
    totalNsys = 0
    totalS = 0
    totalD = 0
    totalI = 0
    totalER = 0
    
    #Segment based
    for segmentID in range(annotatedEventRoll.shape[0]):
        annotatedSegment = annotatedEventRoll[segmentID,:]
        predictedSegment = predictedEventRoll[segmentID,:]
        
        Ntp = np.sum(predictedSegment+annotatedSegment > 1)
        totalNtp += Ntp
        Ntn = np.sum(predictedSegment+annotatedSegment == 0)
        totalNtn += Ntn
        Nfp = np.sum(predictedSegment-annotatedSegment > 0)
        totalNfp += Nfp
        Nfn = np.sum(annotatedSegment-predictedSegment > 0)
        totalNfn += Nfn
        
        Nref = np.sum(annotatedSegment)
        totalNref += Nref
        Nsys = np.sum(predictedSegment)
        totalNsys += Nsys
        
        S = min(Nref,Nsys) - Ntp
        totalS += S
        D = max(0, Nref - Nsys)
        totalD += D
        I = max(0, Nsys - Nref)
        totalI += I
        ER = max(Nref,Nsys) - Ntp
        totalER += ER
     
    """
    #TODO: Add functionality for class wise metrics later
    classMetrics = np.zeros((annotatedEventRoll.shape[1],6))
        
    for classID in range(annotatedEventRoll.shape[1]):
        annotatedSegment = annotatedEventRoll[:,classID]
        predictedSegment = predictedEventRoll[:,classID]
        
        Ntp = np.sum(predictedSegment+annotatedSegment > 1)
        classMetrics[classID,0] += Ntp
        Ntn = np.sum(predictedSegment+annotatedSegment == 0)
        classMetrics[classID,1] += Ntn
        Nfp = np.sum(predictedSegment-annotatedSegment > 0)
        classMetrics[classID,2] += Nfp
        Nfn = np.sum(annotatedSegment-predictedSegment > 0)
        classMetrics[classID,3] += Nfn
        
        Nref = np.sum(annotatedSegment)
        classMetrics[classID,4] += Nref
        Nsys = np.sum(predictedSegment)
        classMetrics[classID,5] += Nsys
    """
    
    return np.array([totalNtp,totalNtn,totalNfp,totalNfn,totalNref,totalNsys,totalS,totalD,totalI,totalER])
    
def aggregation(Ntp,Ntn,Nfp,Nfn,Nref,Nsys,S,D,I,ER):
    prec = Ntp/Nsys
    rec = Ntp/Nref
    F = 2* ((prec * rec) /(prec + rec))
    totalER = ER/Nref
    totalS = S/Nref
    totalD = D/Nref
    totalI = I/Nref
    return [prec,rec,F,totalER,totalS,totalD,totalI]
