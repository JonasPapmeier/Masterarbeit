# -*- coding: utf-8 -*-
"""
Created on Tue May  1 15:40:54 2018

@author: brummli
"""

import librosa
import numpy as np
import copy

def istftWithPhaseEstimation(spectrogram,N_FFT,HOP,num_iteration):
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
    X_t = librosa.istft(X_best,hop_length=HOP,center=True,window='hamming')
    for i in range(num_iteration):
        est = librosa.stft(X_t,n_fft=N_FFT,hop_length=HOP,center=True,window='hamming')
        phase = est / np.maximum(reg, np.abs(est))
        phase = phase[:len(X_s)]
        X_s = X_s[:len(phase)]
        X_best = X_s * phase
        X_t = librosa.istft(X_best,hop_length=HOP,center=True,window='hamming')
    return X_t

#Our test audio file
audioFile = 'testAudio/dev_1_ebr_6_nec_2_poly_0.wav'
#Signal rate
rate = 30000
n_fft = 1024
hop = 512
#Number of bins used in mel spectrograms
mel = 50
#Number of iterations for the Spectrogram inversion algorithm
numIter = 100

maxLengthInSeconds=8.5
maxLengthInFrames=int(maxLengthInSeconds*rate//hop)

#Flags to control which forms should be reconstructed and saved as audio file
outSpec = True
outEnergy = True
outPower = True
outMelSpec = True
outMelEnergy = True
outMelPower = True
outLogMelSpec = True
outLogMelEnergy = True
outLogMelPower = True

#Replaces phase with random phase, necessary as initialization for Griffin-Lim Algorithm
randomizePhase = True

audio, _ = librosa.load(audioFile,sr=rate,mono=True,dtype=np.float32)

#Simple Spec, should be perfectly reconstructed when phase is left untouched
#and no iterations for Griffin-Lim are used
spec = librosa.core.stft(y=audio,n_fft=n_fft,hop_length=hop,center=True,window='hamming')
spec = spec[:,:maxLengthInFrames]
#Generate random phase as initialization for all reconstructions here
randomPhase = np.random.random(size=spec.shape)
if outSpec:
    if randomizePhase:
        invSpec = np.real(spec)*(np.cos(randomPhase)+1j*np.sin(randomPhase))
        invSpec = istftWithPhaseEstimation(invSpec,n_fft,hop,numIter)
    else:
        invSpec = istftWithPhaseEstimation(spec,n_fft,hop,numIter)
    librosa.output.write_wav('testAudio/spec.wav',invSpec,sr=rate,norm=True)

#Absolute is not invertible, phase is lost, should introduce slight error in reconstruction
energySpec = np.abs(spec)
if outEnergy:
    if randomizePhase:
        invEnergySpec = np.real(energySpec)*(np.cos(randomPhase)+1j*np.sin(randomPhase))
        invEnergySpec = istftWithPhaseEstimation(invEnergySpec,n_fft,hop,numIter)
    else:
        invEnergySpec = istftWithPhaseEstimation(energySpec,n_fft,hop,numIter)
    librosa.output.write_wav('testAudio/energySpec.wav',invEnergySpec,sr=rate,norm=True)

#Similar effect on reconstruction as in energy spec
powerSpec = energySpec**2
if outPower:
    invPowerSpec = np.lib.scimath.sqrt(powerSpec)
    if randomizePhase:
        invPowerSpec = np.real(invPowerSpec)*(np.cos(randomPhase)+1j*np.sin(randomPhase))
    invPowerSpec = istftWithPhaseEstimation(invPowerSpec,n_fft,hop,numIter)
    librosa.output.write_wav('testAudio/powerSpec.wav',invPowerSpec,sr=rate,norm=True)

#Mel-filtering of the spectrogram
#Loses information on higher frequencies, these should be lower in reconstruction
melSpec = librosa.feature.melspectrogram(S=spec,sr=rate,n_fft=n_fft,hop_length=hop, n_mels=mel,fmin=100,fmax=15000,htk=True,power=1.0,norm=1)

#To invert mel filtering we want to invert the filter matrix
#as the filter matrix is not square, we can only approximate with a pseudo-inverse
melFilterBank = librosa.filters.mel(rate,n_fft,mel,fmin=100,fmax=15000,htk=True,norm=1)
invMelFilterBank = np.linalg.pinv(melFilterBank)

if outMelSpec:
    invMelSpec = np.dot(invMelFilterBank,melSpec)
    if randomizePhase:
        invMelSpec = np.real(invMelSpec)*(np.cos(randomPhase)+1j*np.sin(randomPhase))
    invMelSpec = istftWithPhaseEstimation(invMelSpec,n_fft,hop,numIter)
    librosa.output.write_wav('testAudio/melSpec.wav',invMelSpec,sr=rate,norm=True)

#Mel-filtering of the energy spec
melEnergySpec = librosa.feature.melspectrogram(S=energySpec,sr=rate,n_fft=n_fft,hop_length=hop, n_mels=mel,fmin=100,fmax=15000,htk=True,power=1.0,norm=1)
if outMelEnergy:
    invMelEnergySpec = np.dot(invMelFilterBank,melEnergySpec)
    if randomizePhase:
        invMelEnergySpec = np.real(invMelEnergySpec)*(np.cos(randomPhase)+1j*np.sin(randomPhase))
    invMelEnergySpec = istftWithPhaseEstimation(invMelEnergySpec,n_fft,hop,numIter)
    librosa.output.write_wav('testAudio/melEnergySpec.wav',invMelEnergySpec,sr=rate,norm=True)

#Mel-filtering of the power spec
#Reconstruction seems to be especially problematic when no iterative algorithm is used
melPowerSpec = librosa.feature.melspectrogram(S=powerSpec,sr=rate,n_fft=n_fft,hop_length=hop, n_mels=mel,fmin=100,fmax=15000,htk=True,power=1.0,norm=1)
if outMelPower:
    invMelPowerSpec = np.dot(invMelFilterBank,melPowerSpec)
    #Use lib.scimath.sqrt to make sure that negative entries are handled correctly
    invMelPowerSpec = np.lib.scimath.sqrt(invMelPowerSpec)
    if randomizePhase:
        invMelPowerSpec = np.real(invMelPowerSpec)*(np.cos(randomPhase)+1j*np.sin(randomPhase))
    invMelPowerSpec = istftWithPhaseEstimation(invMelPowerSpec,n_fft,hop,numIter)
    librosa.output.write_wav('testAudio/melPowerSpec.wav',invMelPowerSpec,sr=rate,norm=True)

#Log scaling of the specs
#Seems perceptually not different
epsilon = 1e-7 #For numerical stability, to avoid log(0)
logMelSpec = np.log(melSpec+epsilon)
if outLogMelSpec:
    invLogMelSpec = np.exp(logMelSpec)
    invLogMelSpec = np.dot(invMelFilterBank,invLogMelSpec)
    if randomizePhase:
        invLogMelSpec = np.real(invLogMelSpec)*(np.cos(randomPhase)+1j*np.sin(randomPhase))
    invLogMelSpec = istftWithPhaseEstimation(invLogMelSpec,n_fft,hop,numIter)
    librosa.output.write_wav('testAudio/logMelSpec.wav',invLogMelSpec,sr=rate,norm=True)

logMelEnergySpec = np.log(melEnergySpec+epsilon)
if outLogMelEnergy:
    invLogMelEnergySpec = np.exp(logMelEnergySpec)
    invLogMelEnergySpec = np.dot(invMelFilterBank,invLogMelEnergySpec)
    if randomizePhase:
        invLogMelEnergySpec = np.real(invLogMelEnergySpec)*(np.cos(randomPhase)+1j*np.sin(randomPhase))
    invLogMelEnergySpec = istftWithPhaseEstimation(invLogMelEnergySpec,n_fft,hop,numIter)
    librosa.output.write_wav('testAudio/logMelEnergySpec.wav',invLogMelEnergySpec,sr=rate,norm=True)

logMelPowerSpec = np.log(melPowerSpec+epsilon)
if outLogMelPower:
    invLogMelPowerSpec = np.exp(logMelPowerSpec)
    invLogMelPowerSpec = np.dot(invMelFilterBank,invLogMelPowerSpec)
    invLogMelPowerSpec = np.lib.scimath.sqrt(invLogMelPowerSpec)
    if randomizePhase:
        invLogMelPowerSpec = np.real(invLogMelPowerSpec)*(np.cos(randomPhase)+1j*np.sin(randomPhase))
    invLogMelPowerSpec = istftWithPhaseEstimation(invLogMelPowerSpec,n_fft,hop,numIter)
    librosa.output.write_wav('testAudio/logMelPowerSpec.wav',invLogMelPowerSpec,sr=rate,norm=True)