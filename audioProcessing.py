# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 17:14:39 2017

@author: brummli
"""

import numpy as np
import librosa
import os
import copy
import config_
from multiprocessing import Pool

def loadSingleFileAudio(fileDescriptionPair):
    folder = fileDescriptionPair[0]
    fileName = fileDescriptionPair[1]
    SR = fileDescriptionPair[2]
    data = (librosa.load(folder+fileName,sr=SR,mono=True,dtype=np.float32,res_type='kaiser_best')[0])
    return (fileName, data)

def loadFolderAudio(folder,SR=16000):
    """
    Loads every wav file in the given folder (no subfolders) into a list
    
    Args:
        folder (str): The folder to be searched for wav files
        SR (int): Sampling rate with which to load the audio data
        
    Returns:
        list: A list containing pairs (name,data) describing each file with its name as a string and the data as numpy array
    """
    files = []
    for fileName in os.listdir(folder):
        if fileName.endswith(('.wav','.mp3','.ogg','.flac')):
            files.append((folder,fileName,SR))
    #processes=None equals to os.cpu_count()
    with Pool(processes=None) as pool:
        filesProcessed = pool.map(loadSingleFileAudio, files)
    return filesProcessed


def istftWithPhaseEstimation(spectrogram,N_FFT=512,HOP=256,center=False, num_iteration=100):
    """
    Transforms Spectrogram into time-series representation. The phase is estimated
    with the iterative Griffin-Lim Algorithm. The phase estimation may be problematic
    for long and for polyphonic audio.
    
    Args:
        spectrogram (np.array): The spectrogram data to be transformed
        N_FFT (int): original number of bins used for the stft
        HOP (int): original hop size used for the stft
        num_iteration (int): number of iterations for phase estimation
        
    Returns:
        np.array: The time-series audio with estimated phase
    """
    print("Estimate Phase")
    X_s = spectrogram
    reg = np.max(X_s) / 1E8
    X_best = copy.deepcopy(X_s)
    X_t = librosa.istft(X_best,hop_length=HOP,center=center,window='hann')
    for i in range(num_iteration):
        est = librosa.stft(X_t,n_fft=N_FFT,hop_length=HOP,center=center,window='hann')
        phase = est / np.maximum(reg, np.abs(est))
        phase = phase[:len(X_s)]
        X_s = X_s[:len(phase)]
        X_best = X_s * phase
        X_t = librosa.istft(X_best,hop_length=HOP,center=center,window='hann')
    return X_t

def augmentParallelHelper(example,sr,timeSteps,pitchSteps,levelSteps,globalNum):
    augmentedExampleList = []
    for step in range(len(levelSteps)):
        levelAdjustedExample = example[1] * levelSteps[step]
        augmentedExampleList.append((example[0],levelAdjustedExample))
    for step in range(len(timeSteps)):
        #Adjust level also for time and pitch shift, so that they aren't related to one fixed level
        #use global number + step, so that for each shift step different level steps are used per example
        level_example = example[1]*levelSteps[(globalNum+step)%len(levelSteps)]
        augmentedExample = librosa.effects.time_stretch(level_example,timeSteps[step])
        augmentedExampleList.append((example[0],augmentedExample))
    for step in range(len(pitchSteps)):
        level_example = example[1]*levelSteps[(globalNum+step)%len(levelSteps)]
        augmentedExample = librosa.effects.pitch_shift(level_example,sr,pitchSteps[step])
        augmentedExampleList.append((example[0],augmentedExample))
    return augmentedExampleList
        

def augment(data,sr,minTime,maxTime,stepsTime,minPitch,maxPitch,stepsPitch,minLevel,maxLevel,stepsLevel):
    """
    Iterates over training data and augments by manipulating time and pitch
    Args:
        data (list): the training data to augment, assumend to be a list containing (label,audio)
        sr (int): the signal rate of data
        minTime (float): minimal time stretch value
        maxTime (float): maximal time stretch value
        stepsTime (int): number of stretches applied between min and max
    Returns:
        list: containing original training data and augmented examples
    """  
    augmentedData = []
    time_stretches = np.linspace(minTime,maxTime,num=stepsTime)
    pitch_shifts = np.linspace(minPitch,maxPitch,num=stepsPitch)
    level_multiplier = 2**np.linspace(minLevel,maxLevel,stepsLevel)
    descriptionPairs = []
    for globalNum, example in enumerate(data):
        descriptionPairs.append((example,sr,time_stretches,pitch_shifts,level_multiplier,globalNum))
    with Pool() as pool:
        augmentedExamplesLists = pool.starmap(augmentParallelHelper,descriptionPairs)
    for examplesList in augmentedExamplesLists:
        augmentedData.extend(examplesList)
    return augmentedData

#TODO: be consistend with naming style for parameters
def melSpecExtraction(data,sr=16000,N_FFT=512,HOP=256,center=True,log_power=True,deriv=True,frameEnergy=True,n_mels=26,max_dB=120):
    """
    Extracts a mel spectrogram. Allows for most additional transformations often performed
    Args:
        data (numpy.array): the raw audio data (1d time signal)
        sr (int): the signal rate of the audio data
        N_FFT (int): number of bins to be used for stft
        HOP=256 (int): hop size to be used for stft
        center (bool): whether beginning and end should be padded with zeros so that the actual 0 index of the audio can be used as center for the first fourier transformation
        log_power (bool): whether the magnitude spectrogram should be transformed to a dB scaled spectrogram by 20*log_10(S**2), where S is the original spectrogram
        deriv (bool): whether the first derivate of all features should be added
        frameEnergy (bool): whether the energy should be added as a feature
        n_mels (int): number of bins for the mel-scaling
    Returns:
        np.array: The extracted features
    """
    spec = librosa.core.stft(y=data,n_fft=N_FFT,hop_length=HOP,center=center,window='hamming')
    spec = np.real(spec * np.conj(spec))
    spec = spec/N_FFT

    spec = librosa.feature.melspectrogram(S=spec,sr=sr,n_fft=N_FFT,hop_length=HOP,
                                          n_mels=n_mels,fmin=50,fmax=sr//2,htk=True,
                                          power=1.0,norm=1)
    if log_power:
        spec = np.log10(spec+np.finfo(float).eps)
    if frameEnergy:
        energy = np.sum(spec, axis=0, keepdims=True)
        spec = np.append(spec,energy,axis=0)
    if deriv:
        deltas = librosa.feature.delta(spec,width=9)
        spec = np.append(spec,deltas,axis=0)
    spec = spec.T
    return spec
    
def invertMelSpecTransformation(melSpec,sr=30000,N_FFT=config_.nFFT,hop=config_.hop,
                            center=True,log_power=config_.logPower,deriv=config_.deriv,
                            frameEnergy=config_.frameEnergy,n_mels=config_.nBin):
    """
    Inverse function for melSpecExtraction, allows to retain audio from a spectrogram.
    """
    #Remove columns that relate to frameEnergy and derivative
    melSpecPart = melSpec[:,:n_mels].T
    if log_power:
        melSpecPart = np.power(10,melSpecPart)
    melFilterBank = librosa.filters.mel(sr,N_FFT,n_mels,fmin=50,fmax=sr//2,htk=True,norm=1)
    inverseMelFilterBank = np.linalg.pinv(melFilterBank)
    powerSpec = np.dot(inverseMelFilterBank,melSpecPart)
    spec = np.lib.scimath.sqrt(powerSpec)
    randomGen = np.random.RandomState(1337)
    randomPhase = randomGen.random_sample(size=spec.shape)
    spec = np.real(spec) *(np.cos(randomPhase)+1j*np.sin(randomPhase))
    return spec
