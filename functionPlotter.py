# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 12:42:07 2018

@author: brummli
"""

import matplotlib as mpl
import latexFormatPlotter
#mpl.use('Agg')
mpl.rcParams.update(latexFormatPlotter.nice_fonts)
import numpy as np
import matplotlib.pyplot as plt

def linePlotFunction(function,start,stop,step,title,xlabel,ylabel,fileNamePostfix=''):
    fig, ax = plt.subplots(1, 1, figsize=latexFormatPlotter.set_size(latexFormatPlotter.textwidthBeamer))
    points = np.arange(start,stop,step)
    transformedPoints = list(map(function,points))
    ax.plot(points,transformedPoints)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    title = title+fileNamePostfix
    plt.savefig('plots/beamer/'+title.replace(' ', '')+'.pdf', format='pdf', bbox_inches='tight')    
    #plt.show()
    plt.close(fig)

def habituationFunction(x):
    alpha = 1.05
    tau = 3.33
    h = 1.0 - (1.0/alpha) * (1 - np.exp(-(alpha*x/tau)))
    return h
    
def activationFunction(x):
    dist = np.square(x-0)
    activation = np.exp(-dist)
    return activation
    
func = habituationFunction
linePlotFunction(func,0,10,0.1,'Habituation Function','Activations','Habituation Value')

linePlotFunction(activationFunction,-3,3,0.01,'Activation Function','Position','Activation')