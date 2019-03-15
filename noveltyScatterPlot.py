# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 13:32:16 2018

@author: brummli
"""

import matplotlib as mpl
import latexFormatPlotter
#mpl.use('Agg')
mpl.rcParams.update(latexFormatPlotter.nice_fonts)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def scatterNoveltyPlotWithLabel(data,labelArray,novelIndex,legend,title,fileNamePostfix=''):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=latexFormatPlotter.set_size(latexFormatPlotter.textwidthBeamer))
    
    colorMap = mpl.cm.get_cmap(name='Paired')
    maxLabel = np.max(labelArray)
    colorsArray = colorMap(labelArray/maxLabel,alpha=1)

    nonNovelData = data[labelArray<novelIndex]
    nonNovelLabels = colorsArray[labelArray<novelIndex]
        
    ax1.scatter(nonNovelData[:,0],nonNovelData[:,1],c=nonNovelLabels,edgecolors='k',s=20)
    ax2.scatter(data[:,0],data[:,1],c=colorsArray,edgecolors='k',s=20)
    ax1.tick_params(axis='both',bottom=False,left=False,labelbottom=False,labelleft=False)
    ax2.tick_params(axis='both',bottom=False,left=False,labelbottom=False,labelleft=False)
    ax2.set_alpha(0.1)
    ax1.set_title('Training time')
    ax2.set_title('Testing time')
    
    title = title+fileNamePostfix
    plt.savefig('plots/beamer/'+title.replace(' ', '')+'.pdf', format='pdf', bbox_inches='tight')    
    #plt.show()
    plt.close(fig)    
    

n_samples = 250
noise = 1.15
X, Y = make_blobs(n_samples=n_samples,n_features=2,centers=5,cluster_std=noise,random_state=42,shuffle=False)
X = X*1

scatterNoveltyPlotWithLabel(X,Y,3,[''],'',fileNamePostfix='NoveltyDetectionExample')
