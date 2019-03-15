# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 21:18:26 2018

@author: brummli
"""

import numpy as np
import matplotlib as mpl
import latexFormatPlotter
mpl.rcParams.update(latexFormatPlotter.nice_fonts)
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import librosa
import dataLoader
import audioProcessing
import config_

sr = 16000

loader = dataLoader.DCASE2dataPrep(SR=sr,dev=True,test=False)
print(len(loader.trainData))
print(len(loader.holdoutData))
minTimeStretch, maxTimeStretch, stepsTimeStretch, minPitchShift, maxPitchShift, stepsPitchShift, minLevelAdjust, maxLevelAdjust, stepsLevelAdjust = config_.getAugmentConfig(config=5)
print('Augment')
aug = audioProcessing.augment(loader.trainData,sr,minTimeStretch,maxTimeStretch,
              stepsTimeStretch,minPitchShift,maxPitchShift,stepsPitchShift,
              minLevelAdjust,maxLevelAdjust,stepsLevelAdjust)
nBin = 99
print('Prepare Train Set')
trainSet = loader.prepareTrainSet(aug,deriv=False,frameEnergy=False,N_FFT=2048,hop=512,n_mels=nBin,lengthScene=10,eventsPerScene=5,lengthExample=10,hopExample=10,asFeatureVector=False,shuffle=False)
#train_x, train_y, train_y_ae = trainSet
devSet = loader.prepareValSet(lengthScene=10,lengthExample=10,
                      hopExample=10,asFeatureVector=False,temporalCompression=False,
                      eventsPerScene=5,N_FFT=2048,hop=512,randSeed=1337,
                      log_power=True,deriv=False,frameEnergy=False,n_mels=nBin,max_dB=120
                      ,relativeLevel=1.0,zNorm=True,minMaxNorm=False,l2Norm=False)
                      
#FOO
print(devSet[0][0].shape,devSet[1][0].shape)

spec = devSet[0][11,0:1000,:].T
labels = devSet[1][11,0:1000,:].T

maxLabel = 5
novelty = np.zeros((1,labels.shape[-1]))
novelty[np.any(labels[maxLabel+1:,:],axis=0,keepdims=True)] = 1

#fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(1600,900))
#fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=latexFormatPlotter.set_size(latexFormatPlotter.textwidthBeamer))
#ax1.pcolor(spec,edgecolor='none',rasterized=True,cmap='magma') #TODO: plot call on ax object instead?
#ax2.pcolor(labels,edgecolor='none',rasterized=True)
#vline1 = ax1.axvline(x=0)
#vline2 = ax2.axvline(x=0)
#plt.savefig(basefile+str(num)+'.png')
#ax1.set_ylabel('Mel Bins')
#ax2.set_ylabel('Event Label')
#ax2.set_xlabel('Frames')
#plt.savefig('plots/beamer/fullLength.pdf', format='pdf', bbox_inches='tight')
#plt.close(fig)

#actualEvents = np.nonzero(labels[0:5,:])
#labels[actualEvents] = labels[actualEvents] + 1
labels[0:maxLabel+1,:] = labels[0:maxLabel+1,:] + 0.25
fig, ax2 = plt.subplots(1, 1, figsize=latexFormatPlotter.set_size(latexFormatPlotter.textwidthBeamer))
#ax1.pcolor(spec,edgecolor='none',rasterized=True,cmap='magma') #TODO: plot call on ax object instead?
ax2.pcolor(labels,edgecolor='none',rasterized=True)
#vline1 = ax1.axvline(x=0)
#vline2 = ax2.axvline(x=0)
#plt.savefig(basefile+str(num)+'.png')
#ax1.set_ylabel('Mel Bins')
ax2.set_ylabel('Event Label')
ax2.set_xlabel('Frames')
plt.savefig('plots/beamer/withHalfIncreased.pdf', format='pdf', bbox_inches='tight')
plt.close(fig)


fig, ax2 = plt.subplots(1, 1, figsize=latexFormatPlotter.set_size(latexFormatPlotter.textwidthBeamer))
#ax1.pcolor(spec,edgecolor='none',rasterized=True,cmap='magma') #TODO: plot call on ax object instead?
ax2.pcolor(np.concatenate((labels[0:maxLabel+1,:],novelty),axis=0),edgecolor='none',rasterized=True)
#vline1 = ax1.axvline(x=0)
#vline2 = ax2.axvline(x=0)
#plt.savefig(basefile+str(num)+'.png')
#ax1.set_ylabel('Mel Bins')
ax2.set_ylabel('Event Label')
ax2.set_xlabel('Frames')
plt.savefig('plots/beamer/withNovelty.pdf', format='pdf', bbox_inches='tight')
plt.close(fig)


fig, ax2 = plt.subplots(1, 1, figsize=latexFormatPlotter.set_size(latexFormatPlotter.textwidthBeamer))
#ax1.pcolor(spec,edgecolor='none',rasterized=True,cmap='magma') #TODO: plot call on ax object instead?
ax2.pcolor(labels[:,np.squeeze(novelty==0)],edgecolor='none',rasterized=True)
#vline1 = ax1.axvline(x=0)
#vline2 = ax2.axvline(x=0)
#plt.savefig(basefile+str(num)+'.png')
#ax1.set_ylabel('Mel Bins')
ax2.set_ylabel('Event Label')
ax2.set_xlabel('Frames')
plt.savefig('plots/beamer/withOnlyIncremental.pdf', format='pdf', bbox_inches='tight')
plt.close(fig)
exit()

frameRateAnim = 30.
interval = 1000./frameRateAnim

def animate(i):
    #i should be the frame number
    frameNumber = librosa.core.time_to_frames([i/frameRateAnim],sr=sr,hop_length=512,n_fft=2048)[0]
    vline1.set_xdata([frameNumber,frameNumber])
    vline2.set_xdata([frameNumber,frameNumber])
    
anim = FuncAnimation(fig, animate, interval=interval,frames=1000)
anim.save('plots/beamer/test.mp4',bitrate=-1,codec="libx264",dpi=300)
plt.show()