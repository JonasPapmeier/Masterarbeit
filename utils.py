# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 15:50:54 2018

@author: brummli
"""

import matplotlib as mpl
import latexFormatPlotter
#mpl.use('Agg')
#mpl.rcParams.update(latexFormatPlotter.nice_fonts)
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap,TSNE


def logger(fileName, string):
    with open(fileName,'a') as f:
        f.write(string)
    f.close()
    
    
def plotAsPcolor(data,basefile,maxNum,ylabel='Mel-bin'):
    for num, example in enumerate(data):
        fig, ax = plt.subplots(1, 1, figsize=latexFormatPlotter.set_size(latexFormatPlotter.textwidthThesis), num=num)
        plt.pcolormesh(example.T,edgecolor='none',rasterized=True,cmap='magma') 
        ax.axvline(x=50)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Frames')
        plt.savefig(basefile+str(num)+'.pdf', format='pdf', bbox_inches='tight')
        plt.close(fig)
        if num>maxNum:
            break
        
def plotAsLinePlot(data,basefile,maxNum):
    for num, example in enumerate(data):
        fig, ax = plt.subplots(1, 1, figsize=latexFormatPlotter.set_size(latexFormatPlotter.textwidthThesis), num=num)
        plt.plot(example.T)
        plt.savefig(basefile+str(num)+'.pdf', format='pdf', bbox_inches='tight')
        plt.close(fig)
        if num > maxNum:
            break
        
def reduceDimensions(dataArray,method=None,perplexity=30,learning_rate=200):
    if method == 'Isomap':
        reduceModel = Isomap(n_components=2)
    elif method == 'TSNE':
        reduceModel = TSNE(n_components=2,perplexity=perplexity,verbose=1,learning_rate=learning_rate,n_iter=1000,n_iter_without_progress=200,min_grad_norm=1e-7,early_exaggeration=12.0,method='barnes_hut')
    elif method == 'PCA':
        reduceModel = PCA(n_components=2)
    else:
        return dataArray
    reducedArray = reduceModel.fit_transform(dataArray)
    return reducedArray
    
def prepareDataForPlotting(x,y,skips):
    prepared_x = np.reshape(x,(-1,x.shape[-1]))
    prepared_x = prepared_x[::skips,:]
    #Find and remove duplicates as they can influence TSNE algorithm
    prepared_x, uniqueIndices = np.unique(prepared_x,return_index=True,axis=0)

    prepared_y = np.reshape(y,(-1,y.shape[-1]))
    prepared_y = prepared_y[::skips,:]
    prepared_y = prepared_y[uniqueIndices]
    prepared_y = np.concatenate((np.zeros((prepared_y.shape[0],1)),prepared_y),axis=-1)
    prepared_y = np.argmax(prepared_y,axis=-1)
    return (prepared_x,prepared_y)

def plotHelper(fileName,dataArray,labelArray):
    if len(dataArray.shape) != 2:
        print('Please reshape the data correctly. Your data shape is:', dataArray.shape)
    elif dataArray.shape[-1] > 3 or dataArray.shape[-1] < 2:
        print('Please reduce data dimensions to 2 or 3 dimensions. Your data has dimension:', dataArray.shape[-1])
    elif dataArray.shape[-1] == 3:
        plot3DHelper(fileName,dataArray,labelArray)
    elif dataArray.shape[-1] == 2:
        plot2DHelper(fileName,dataArray,labelArray)
    else:
        print('Something went totally wrong during plotting')
 
def plot2DHelper(fileName,dataArray,labelArray):
    fig = plt.figure()
    colorMap = mpl.cm.get_cmap(name='Paired')
    maxLabel = np.max(labelArray)
    colorsArray = colorMap(labelArray/maxLabel,alpha=1)
    plt.scatter(dataArray[:,0],dataArray[:,1],c=colorsArray,edgecolors='k')
    classes = range(int(maxLabel)+1)
    classNames = ['background','clearthroat','cough','doorslam', 'drawer', 'keyboard', 'keys', 'knock', 'laughter', 'pageturn', 'phone', 'speech']
    class_colours = list(map(colorMap,map(lambda x: x/(maxLabel*1.0),classes)))
    recs = []
    for i in range(0,len(class_colours)):
        recs.append(mpl.patches.Rectangle((0,0),1,1,fc=class_colours[i]))
    plt.legend(recs,classes,loc='best').get_frame().set_alpha(0.5)
    plt.savefig(fileName)
    plt.close(fig)
       
def plot3DHelper(fileName,dataArray,labelArray):
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.view_init(7, -80)
    colorMap = mpl.cm.get_cmap(name='Set1')
    maxLabel = np.max(labelArray)
    colorsArray = colorMap(labelArray/maxLabel,alpha=1)
    ax.scatter(dataArray[:,0], dataArray[:,1], dataArray[:,2],c=colorsArray)
    classes = range(int(maxLabel)+1)
    class_colours = list(map(colorMap,map(lambda x: x/(maxLabel*1.0),classes)))
    recs = []
    for i in range(0,len(class_colours)):
        recs.append(mpl.patches.Rectangle((0,0),1,1,fc=class_colours[i]))
    plt.legend(recs,classes,loc='best').get_frame().set_alpha(0.5)
    plt.savefig(fileName)
    plt.close(fig)
   
#Taken from:
#https://gist.github.com/nils-werner/9d321441006b112a4b116a8387c2280c     
#Last accessed: 04.04.2018, 18:12
#TODO: padded has no current actual meaning, so using non fitting size and stepsize leads to access on memory outside of the array
def sliding_window(data, size, stepsize=1, padded=False, axis=-1, copy=True):
    """
    Calculate a sliding window over a signal
    Parameters
    ----------
    data : numpy array
        The array to be slided over.
    size : int
        The sliding window size
    stepsize : int
        The sliding window stepsize. Defaults to 1.
    axis : int
        The axis to slide over. Defaults to the last axis.
    copy : bool
        Return strided array as copy to avoid sideffects when manipulating the
        output array.
    Returns
    -------
    data : numpy array
        A matrix where row in last dimension consists of one instance
        of the sliding window.
    Notes
    -----
    - Be wary of setting `copy` to `False` as undesired sideffects with the
      output values may occurr.
    Examples
    --------
    >>> a = numpy.array([1, 2, 3, 4, 5])
    >>> sliding_window(a, size=3)
    array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])
    >>> sliding_window(a, size=3, stepsize=2)
    array([[1, 2, 3],
           [3, 4, 5]])
    See Also
    --------
    pieces : Calculate number of pieces available by sliding
    """
    if axis >= data.ndim:
        raise ValueError(
            "Axis value out of range"
        )

    if stepsize < 1:
        raise ValueError(
            "Stepsize may not be zero or negative"
        )

    if size > data.shape[axis]:
        raise ValueError(
            "Sliding window size may not exceed size of selected axis"
        )

    shape = list(data.shape)
    shape[axis] = np.floor(data.shape[axis] / stepsize - size / stepsize + 1).astype(int)
    shape.append(size)

    strides = list(data.strides)
    strides[axis] *= stepsize
    strides.append(data.strides[axis])

    strided = np.lib.stride_tricks.as_strided(
        data, shape=shape, strides=strides
    )

    if copy:
        return strided.copy()
    else:
        return strided
