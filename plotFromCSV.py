# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 16:14:55 2018

@author: brummli
"""


import matplotlib as mpl
import latexFormatPlotter
#mpl.use('Agg')
mpl.rcParams.update(latexFormatPlotter.nice_fonts)
import numpy as np
import matplotlib.pyplot as plt
import itertools

def loadCSV(filePath):
    data = np.genfromtxt(filePath,delimiter=',',names=True, dtype=None)#dtype=("|S10", "|S10", float, float, float,float,float,float,float))
    return data
    
def plotAsBarReprOld(data,xLabeling,hidLegend,title,std=None,numberPoints=4,fileNamePostfix=''):
    fig, ax = plt.subplots(1, 1, figsize=latexFormatPlotter.set_size(latexFormatPlotter.textwidthBeamer))
    ind = np.arange(len(data))+1
    #transform to percentage
    data = np.round(data*100,decimals=1)
    if np.any(std):
        std = np.round(std*100,decimals=1)
    rects1 = ax.bar(ind,data,yerr=std)
    ax.set_xticks(ind)
    ax.set_xticklabels(xLabeling,minor=False)
    #colors=['r','y','g']
    colors = mpl.cm.get_cmap(name='Paired')#.to_rgba(np.arange(3))
    for i, rect in enumerate(rects1):
        if hidLegend:
            if len(hidLegend) < 3:
                colorIndex = (i+numberPoints)//numberPoints
            else:
                colorIndex = i//numberPoints
        else:
            colorIndex = i//numberPoints
        rect.set_facecolor(colors(colorIndex))
    ax.set_ylim([25,90])
    ax.set_xlabel(r'Mel Bins')
    ax.set_ylabel(r'F1\%')
    ax.set_title(title)
    if hidLegend:
        legIndices = list(map(int,np.linspace(0,len(rects1),num=len(hidLegend),endpoint=False)))
        print(legIndices)
        legList = []
        for ind in legIndices:
            legList.append(rects1[ind])
        ax.legend(legList,hidLegend)
    
    title = title+fileNamePostfix
    plt.savefig('plots/beamer/'+title.replace(' ', '')+'.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig)

def plotAsBarRepr(data,xLabeling,hidLegend,title,std=None,numberPoints=4,fileNamePostfix=''):
    fig, ax = plt.subplots(1, 1, figsize=latexFormatPlotter.set_size(latexFormatPlotter.textwidthBeamer))
    #transform to percentage
    data = np.round(data*100,decimals=1)
    if np.any(std):
        std = np.round(std*100,decimals=1)
    totalGroups = len(data)/numberPoints
    print(totalGroups)
    totalGroups = int(totalGroups)
    data = np.reshape(data,(totalGroups,-1))
    std = np.reshape(std,(totalGroups,-1))
    width=0.27
    allRects = []
    for ind, subData in enumerate(data):
        indices = np.arange(numberPoints)+1
        rects = ax.bar(indices+(ind-1)*width,subData,width,yerr=std[ind])
        allRects.append(rects)
    ax.set_xticks(indices)
    ax.set_xticklabels(xLabeling,minor=False)
    colors = mpl.cm.get_cmap(name='Paired')
    for j, rects in enumerate(allRects):
        for i, rect in enumerate(rects):
            colorIndex = j
            rect.set_facecolor(colors(colorIndex))
    ax.set_ylim([0,100])
    ax.set_xlabel(r'Mel Bins')
    ax.set_ylabel(r'F1\%')
    ax.set_title(title)
    if hidLegend:
        legIndices = list(map(int,np.linspace(0,len(allRects),num=len(hidLegend),endpoint=False)))
        print(legIndices)
        legList = []
        for ind in legIndices:
            legList.append(allRects[ind][0])
        ax.legend(legList,hidLegend)
    
    title = title+fileNamePostfix
    plt.savefig('plots/beamer/'+title.replace(' ', '')+'.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig)


def plotAsBarGWR(data,xLabeling,xLabel,hidLegend,title,std=None,numberPoints=4,fileNamePostfix=''):
    fig, ax = plt.subplots(1, 1, figsize=latexFormatPlotter.set_size(latexFormatPlotter.textwidthBeamer))
    ind = np.arange(len(data))+1
    #transform to percentage
    data = np.round(data*100,decimals=1)
    if np.any(std):
        std = np.round(std*100,decimals=1)
    rects1 = ax.bar(ind,data,yerr=std)
    ax.set_xticks(ind)
    ax.set_xticklabels(xLabeling,minor=False)
    colors = mpl.cm.get_cmap(name='Paired')
    for i, rect in enumerate(rects1):
        if hidLegend:
            if len(hidLegend) < 3:
                colorIndex = (i+numberPoints)//numberPoints
            else:
                colorIndex = i//numberPoints
        else:
            colorIndex = i//numberPoints
        rect.set_facecolor(colors(colorIndex))
    ax.set_ylim([0,100])
    ax.set_xlabel(xLabel)
    ax.set_ylabel(r'F1\%')
    ax.set_title(title)
    if hidLegend:
        legIndices = list(map(int,np.linspace(0,len(rects1),num=len(hidLegend),endpoint=False)))
        print(legIndices)
        legList = []
        for ind in legIndices:
            legList.append(rects1[ind])
        ax.legend(legList,hidLegend)
    
    title = title+fileNamePostfix
    plt.savefig('plots/beamer/'+title.replace(' ', '')+'.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig)
    
def plotAsHorizontalBar(data,xLabeling,xLabel,hidLegend,title,std=None,numberPoints=4,fileNamePostfix=''):
    fig, ax = plt.subplots(1, 1, figsize=latexFormatPlotter.set_size(latexFormatPlotter.textwidthBeamer))
    ind = np.arange(len(data))+1
    data = np.round(data*100,decimals=1)
    if np.any(std):
        std = np.round(std*100,decimals=1)
    rects1 = ax.barh(ind,data,xerr=std)
    ax.set_yticks(ind)
    ax.set_yticklabels(xLabeling,minor=False)
    colors = mpl.cm.get_cmap(name='Paired')
    for i, rect in enumerate(rects1):
        if hidLegend:
            if len(hidLegend) < 3:
                colorIndex = (i+numberPoints)//numberPoints
            else:
                colorIndex = i//numberPoints
        else:
            colorIndex = i//numberPoints
        rect.set_facecolor(colors(colorIndex))
    ax.set_xlim([0,100])
    ax.set_ylabel(xLabel)
    ax.invert_yaxis()
    ax.set_xlabel(r'F1\%')
    ax.set_title(title)
    if hidLegend:
        legIndices = list(map(int,np.linspace(0,len(rects1),num=len(hidLegend),endpoint=False)))
        print(legIndices)
        legList = []
        for ind in legIndices:
            legList.append(rects1[ind])
        ax.legend(legList,hidLegend)
    
    title = title+fileNamePostfix
    plt.savefig('plots/beamer/'+title.replace(' ', '')+'.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig)

    
def plotAsLineInc(dataList,labeling,title,std=None,fileNamePostfix=''):
    fig, ax = plt.subplots(1, 1, figsize=latexFormatPlotter.set_size(latexFormatPlotter.textwidthBeamer))
    markers=itertools.cycle(('--s', '--x', '--^', '--o', '--*'))
    for num, data in enumerate(dataList):
        data_y = data
        data_x = np.arange(len(data))+1
        data_y = np.round(data_y*100,decimals=1)
        ax.plot(data_x, data_y, next(markers), label=labeling[num])
        if std:
            curStd = std[num]
            curStd = np.round(curStd*100,decimals=1)
            print(curStd)
            ax.fill_between(data_x, data_y-curStd, data_y+curStd, alpha=0.3)
    ax.set_ylim([0,100])
    ax.set_xlabel(r'Class Increment')
    ax.set_ylabel(r'F1\%')
    ax.set_title(title)
    ax.legend().get_frame().set_alpha(0.5)
    
    title = title+fileNamePostfix
    plt.savefig('plots/beamer/'+title.replace(' ', '')+'.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig)
    
def plotAsLineNodeF1(dataList,labeling,title,fileNamePostfix=''):
    fig, ax = plt.subplots(1, 1, figsize=latexFormatPlotter.set_size(latexFormatPlotter.textwidthBeamer))
    markers=itertools.cycle(('--s', '--x', '--^', '--o', '--*'))
    for num, data in enumerate(dataList):
        data_x, data_y = data
        data_y = np.round(data_y*100,decimals=1)
        ax.plot(data_x, data_y, next(markers), label=labeling[num])
    ax.set_ylim([20,100])
    ax.set_xlabel(r'Number Nodes')
    ax.set_ylabel(r'F1\%')
    ax.set_title(title)
    ax.legend()
    
    title = title+fileNamePostfix
    plt.savefig('plots/beamer/'+title.replace(' ', '')+'.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig)
    
    
def transformToMeanStd(data,repeats):
    reshaped = np.reshape(data,(-1,repeats))
    mean = np.mean(reshaped,axis=-1)
    std = np.std(reshaped,axis=-1)
    print(mean,std)
    print('Average mean {}, average std {}'.format(np.nanmean(mean),np.nanmean(std)))
    return mean, std
    
if __name__ == '__main__':
    
    #"""
    print('Artificial Noise')
    foo = loadCSV('plots/thesisReady/data/ArtificialNoiseRNN.csv')
    processed, processedStd = transformToMeanStd(foo['F1'],5)
    plotAsBarRepr(processed,[50,80,120,200]*3,['Hidden dim: 60','Hidden dim: 80','Hidden dim: 100'],'RNN Denoise Artificial',std=processedStd)

    foo = loadCSV('plots/thesisReady/data/ArtificialNoiseSeq2Seq.csv')
    processed, processedStd = transformToMeanStd(foo['F1'],5)
    plotAsBarRepr(processed,[50,80,120,200]*3,['Hidden dim: 60','Hidden dim: 80','Hidden dim: 100'],'Seq2Seq Denoise Artificial',std=processedStd)    
    
    foo = loadCSV('plots/thesisReady/data/ArtificialNoiseFFAE.csv')
    processed, processedStd = transformToMeanStd(foo['F1'],5)
    plotAsBarRepr(processed,[50,80,120]*3,['Hidden dim: 60','Hidden dim: 80','Hidden dim: 100'],'FF Denoise Artificial',std=processedStd,numberPoints=3)
    
    foo = loadCSV('plots/thesisReady/data/ArtificialNoiseSeq2SeqLength.csv')
    processed, processedStd = transformToMeanStd(foo['F1'],5)
    plotAsBarGWR(processed,[5,10,11,15,20,30]*1,'Length of input window',['Hidden dim: 100'],'Lenght of input: Seq2Seq Denoise Artificial',std=processedStd,numberPoints=6)    
    #"""    
    
    
    #"""
    print('No Noise')
    foo = loadCSV('plots/thesisReady/data/NoNoiseRNN.csv')
    processed, processedStd = transformToMeanStd(foo['F1'],5)
    plotAsBarRepr(processed,[50,80,120,200]*3,['Hidden dim: 60','Hidden dim: 80','Hidden dim: 100'],'RNN Autoencoder',std=processedStd)
    
    foo = loadCSV('plots/thesisReady/data/NoNoiseSeq2Seq.csv')
    processed, processedStd = transformToMeanStd(foo['F1'],5)
    plotAsBarRepr(processed,[50,80,120,200]*3,['Hidden dim: 60','Hidden dim: 80','Hidden dim: 100'],'Seq2Seq Autoencoder',std=processedStd)
    
    foo = loadCSV('plots/thesisReady/data/NoNoiseFFAE.csv')
    processed, processedStd = transformToMeanStd(foo['F1'],5)
    plotAsBarRepr(processed,[50,80,120]*3,['Hidden dim: 60','Hidden dim: 80','Hidden dim: 100'],'FF Autoencoder',std=processedStd,numberPoints=3)
    #"""
    
    #"""
    print('Dropout Noise')
    foo = loadCSV('plots/thesisReady/data/DropoutNoiseRNN.csv')
    processed, processedStd = transformToMeanStd(foo['F1'],5)
    plotAsBarRepr(processed,[50,120,200]*2,['Hidden dim: 80','Hidden dim: 100'],'RNN Denoise Dropout',std=processedStd,numberPoints=3)

    foo = loadCSV('plots/thesisReady/data/DropoutNoiseSeq2Seq.csv')
    processed, processedStd = transformToMeanStd(foo['F1'],5)
    plotAsBarRepr(processed,[50,120,200]*2,['Hidden dim: 80','Hidden dim: 100'],'Seq2Seq Denoise Dropout',std=processedStd,numberPoints=3)
    
    foo = loadCSV('plots/thesisReady/data/DropoutNoiseFFAE.csv')
    processed, processedStd = transformToMeanStd(foo['F1'],5)
    plotAsBarRepr(processed,[50,120]*2,['Hidden dim: 80','Hidden dim: 100'],'FF Denoise Dropout',std=processedStd,numberPoints=2)
    #"""    
    
    """
    print('ESC')
    #TOOD: this plot needs to be improved
    foo = loadCSV('plots/thesisReady/data/ESCPreTrain.csv')
    processed, processedStd = transformToMeanStd(foo['F1'],5)
    plotAsBarRepr(processed,['Full','No noise class','no class overlap','dropout']*1,None,'Training autoencoder on ESC-50',std=processedStd)
    """
    
    #"""
    print('GWR')
    #foo = loadCSV('plots/thesisReady/data/GWRInsThres.csv')
    #plotAsBarGWR(foo['F1'],[0.7,0.8,0.9]*2,r'Insertion Threshold',['Mel-bins: 25','Mel-bins: 50'],'GWR Raw Input: Insertion Threshold',numberPoints=3)
    gwr1 = loadCSV('plots/thesisReady/data/GWRInsThres50bin.csv')
    gwr2 = loadCSV('plots/thesisReady/data/GWRInsThres200bin.csv')
    gwr3 = loadCSV('plots/thesisReady/data/GWRSeq2Seq100-200.csv')
    gwr4 = loadCSV('plots/thesisReady/data/GWRRNN100-200.csv')
    gwr5 = loadCSV('plots/thesisReady/data/GWRSeq2Seq100-50Artificial.csv')
    combined = [(gwr1['Nodes'],gwr1['F1']),(gwr2['Nodes'],gwr2['F1']),(gwr3['Nodes'],gwr3['F1']),(gwr4['Nodes'],gwr4['F1']),(gwr5['Nodes'],gwr5['F1'])]
    plotAsLineNodeF1(combined,['50 Mel-Bins','200 Mel-Bins','Seq2Seq-Dropout','RNN-Dropout','Seq2Seq-Artificial'],'',fileNamePostfix='GWR Variants')
    #"""
    
    #"""
    foo = loadCSV('plots/thesisReady/data/GWRMaxAge.csv')
    plotAsBarGWR(foo['F1'],[5,10,20,50]*1,r'Max Age',None,'GWR Raw Input: Max Age',numberPoints=4)
    #"""
    
    print('Incremental stuff')
    #foo = loadCSV('plots/thesisReady/data/IncNovelGWR.csv')
    #plotAsLineInc([foo['F1'],foo['IncF1']],['Class','Novel'],'GWR Novelty with Incremental Training')
    
    """
    foo = loadCSV('plots/thesisReady/data/IncNovelGWR8-775Test.csv')
    processed, processedStd = transformToMeanStd(foo['F1'],3)
    plotAsBarGWR(processed,[i for i in range(11)]*1,r'Increment',None,'GWR Ins 0_8 beta 0_775 F1',std=processedStd,numberPoints=11)
    processed, processedStd = transformToMeanStd(foo['NovF1'],3)
    plotAsBarGWR(processed,[i for i in range(11)]*1,r'Increment',None,'GWR Ins 0_8 beta 0_775 F1 Novel',std=processedStd,numberPoints=11)
    processed, processedStd = transformToMeanStd(foo['IncF1'],3)
    plotAsBarGWR(processed,[i for i in range(11)]*1,r'Increment',None,'GWR Ins 0_8 beta 0_775 F1 Incremental',std=processedStd,numberPoints=11)
    foo = loadCSV('plots/thesisReady/data/IncNovelGWR8-775DevNoReject.csv')  
    processed, processedStd = transformToMeanStd(foo['F1'],3)
    plotAsBarGWR(processed,[i for i in range(11)]*1,r'Increment',None,'GWR Ins 0_8 beta 0_0 F1',std=processedStd,numberPoints=11)
    processed, processedStd = transformToMeanStd(foo['NovF1'],3)
    #plotAsBarGWR(processed,[i for i in range(11)]*1,r'Increment',None,'GWR Ins 0_8 beta 0_0 F1 Novel',std=processedStd,numberPoints=11)
    processed, processedStd = transformToMeanStd(foo['IncF1'],3)
    plotAsBarGWR(processed,[i for i in range(11)]*1,r'Increment',None,'GWR Ins 0_8 beta 0_0 F1 Incremental',std=processedStd,numberPoints=11)
    foo = loadCSV('plots/thesisReady/data/IncNovelGWR8-775Test.csv')
    processed, processedStd = transformToMeanStd(foo['F1'],3)
    plotAsBarGWR(processed,[i for i in range(11)]*1,r'Increment',None,'Test GWR Ins 0_8 beta 0_775 F1',std=processedStd,numberPoints=11)
    processed, processedStd = transformToMeanStd(foo['NovF1'],3)
    plotAsBarGWR(processed,[i for i in range(11)]*1,r'Increment',None,'Test GWR Ins 0_8 beta 0_775 F1 Novel',std=processedStd,numberPoints=11)
    processed, processedStd = transformToMeanStd(foo['IncF1'],3)
    plotAsBarGWR(processed,[i for i in range(11)]*1,r'Increment',None,'Test GWR Ins 0_8 beta 0_775 F1 Incremental',std=processedStd,numberPoints=11)
    foo = loadCSV('plots/thesisReady/data/IncNovelGWR8-775TestNoReject.csv')
    processed, processedStd = transformToMeanStd(foo['F1'],3)
    plotAsBarGWR(processed,[i for i in range(11)]*1,r'Increment',None,'Test GWR Ins 0_8 beta 0_0 F1',std=processedStd,numberPoints=11)
    processed, processedStd = transformToMeanStd(foo['NovF1'],3)
    #plotAsBarGWR(processed,[i for i in range(11)]*1,r'Increment',None,'GWR Ins 0_8 beta 0_775 F1 Novel',std=processedStd,numberPoints=11)
    processed, processedStd = transformToMeanStd(foo['IncF1'],3)
    plotAsBarGWR(processed,[i for i in range(11)]*1,r'Increment',None,'Test GWR Ins 0_8 beta 0_0 F1 Incremental',std=processedStd,numberPoints=11)
    """
    
    #"""
    foo = loadCSV('plots/thesisReady/data/IncNovelGWR825-8Dev.csv')
    print('Ours')
    processed, processedStd = transformToMeanStd(foo['F1'],3)
    plotAsBarGWR(processed,['foo'*i for i in range(1,12)]*1,r'Increment',None,'',fileNamePostfix='Incremental learning with unseen classes',std=processedStd,numberPoints=11)
    #plotAsHorizontalBar(processed,['foo'*i for i in range(1,12)]*1,r'Increment',None,'',fileNamePostfix='Incremental learning with unseen classes',std=processedStd,numberPoints=11)
    print('Novelty')
    processed, processedStd = transformToMeanStd(foo['NovF1'],3)
    plotAsBarGWR(processed,[i for i in range(1,11)]*1,r'Increment',None,'',fileNamePostfix='Detection of novel classes',std=processedStd,numberPoints=10)
    #processed, processedStd = transformToMeanStd(foo['IncF1'],3)
    #plotAsBarGWR(processed,[i for i in range(1,12)]*1,r'Increment',None,'Incremental learning',std=processedStd,numberPoints=11)
    foo = loadCSV('plots/thesisReady/data/IncNovelGWR825-8DevNoReject.csv')
    print('Non-reject')
    processed, processedStd = transformToMeanStd(foo['F1'],3)
    plotAsBarGWR(processed,[i for i in range(1,12)]*1,r'Increment',None,'',fileNamePostfix='Incremental learning with unseen classes: No detection',std=processedStd,numberPoints=11)
    #processed, processedStd = transformToMeanStd(foo['NovF1'],3)
    #plotAsBarGWR(processed,[i for i in range(11)]*1,r'Increment',None,'GWR Ins 0_825 beta 0_8 F1 Novel',std=processedStd,numberPoints=11)
    print('Incremental')
    processed, processedStd = transformToMeanStd(foo['IncF1'],3)
    plotAsBarGWR(processed,[i for i in range(1,12)]*1,r'Increment',None,'',fileNamePostfix='Incremental learning',std=processedStd,numberPoints=11)
    foo = loadCSV('plots/thesisReady/data/IncNovelGWR825-8Test.csv')
    print('Test Ours')
    processed, processedStd = transformToMeanStd(foo['F1'],3)
    plotAsBarGWR(processed,[i for i in range(1,12)]*1,r'Increment',None,'',fileNamePostfix='Test set: Incremental learning with unseen classes',std=processedStd,numberPoints=11)
    print('Novelty')
    processed, processedStd = transformToMeanStd(foo['NovF1'],3)
    plotAsBarGWR(processed,[i for i in range(1,11)]*1,r'Increment',None,'',fileNamePostfix='Test set: Detection of novel classes',std=processedStd,numberPoints=10)
    #processed, processedStd = transformToMeanStd(foo['IncF1'],3)
    #plotAsBarGWR(processed,[i for i in range(1,12)]*1,r'Increment',None,'Test set: Incremental learning',std=processedStd,numberPoints=11)
    foo = loadCSV('plots/thesisReady/data/IncNovelGWR825-8TestNoReject.csv')
    print('Non-reject')
    processed, processedStd = transformToMeanStd(foo['F1'],3)
    plotAsBarGWR(processed,[i for i in range(1,12)]*1,r'Increment',None,'',fileNamePostfix='Test set: Incremental learning with unseen classes: No detection',std=processedStd,numberPoints=11)
    #processed, processedStd = transformToMeanStd(foo['NovF1'],3)
    #plotAsBarGWR(processed,[i for i in range(11)]*1,r'Increment',None,'GWR Ins 0_825 beta 0_8 F1 Novel',std=processedStd,numberPoints=11)
    print('Incremental')
    processed, processedStd = transformToMeanStd(foo['IncF1'],3)
    plotAsBarGWR(processed,[i for i in range(1,12)]*1,r'Increment',None,'',fileNamePostfix='Test set: Incremental learning',std=processedStd,numberPoints=11)
    #"""    
    
    withReject = loadCSV('plots/thesisReady/data/IncNovelGWR825-8Dev.csv')    
    noReject = loadCSV('plots/thesisReady/data/IncNovelGWR825-8DevNoReject.csv')
    method, methodStd = transformToMeanStd(withReject['F1'],3)
    inc, incStd = transformToMeanStd(noReject['IncF1'],3)
    fail, failStd = transformToMeanStd(noReject['F1'],3)
    combined=[method,inc,fail]
    combinedStd=[methodStd,incStd,failStd]
    plotAsLineInc(combined,['Our approach with detection','Incremental','Our approach without detection'],'',fileNamePostfix='Incremental Learning and the effect of unseen classes',std=combinedStd)
    
    withReject = loadCSV('plots/thesisReady/data/IncNovelGWR825-8Test.csv')    
    noReject = loadCSV('plots/thesisReady/data/IncNovelGWR825-8TestNoReject.csv')
    method, methodStd = transformToMeanStd(withReject['F1'],3)
    inc, incStd = transformToMeanStd(noReject['IncF1'],3)
    fail, failStd = transformToMeanStd(noReject['F1'],3)
    combined=[method,inc,fail]
    combinedStd=[methodStd,incStd,failStd]
    plotAsLineInc(combined,['Our approach with detection','Incremental','Our approach without detection'],'',fileNamePostfix='Test set: Incremental Learning and the effect of unseen classes_Test',std=combinedStd)
    
    #"""
    print('With insertion 775 and threshold 775')
    withReject = loadCSV('plots/thesisReady/data/IncNovelGWR775-775Dev.csv')    
    noReject = loadCSV('plots/thesisReady/data/IncNovelGWR775-775DevNoReject.csv')
    method, methodStd = transformToMeanStd(withReject['F1'],3)
    inc, incStd = transformToMeanStd(noReject['IncF1'],3)
    fail, failStd = transformToMeanStd(noReject['F1'],3)
    combined=[method,inc,fail]
    combinedStd=[methodStd,incStd,failStd]
    plotAsLineInc(combined,['Our approach with detection','Incremental','Our approach without detection'],'Incremental Learning and the effect of unseen classes',fileNamePostfix='_0775',std=combinedStd)
    
    withReject = loadCSV('plots/thesisReady/data/IncNovelGWR775-775Test.csv')    
    noReject = loadCSV('plots/thesisReady/data/IncNovelGWR775-775TestNoReject.csv')
    method, methodStd = transformToMeanStd(withReject['F1'],3)
    inc, incStd = transformToMeanStd(noReject['IncF1'],3)
    fail, failStd = transformToMeanStd(noReject['F1'],3)
    combined=[method,inc,fail]
    combinedStd=[methodStd,incStd,failStd]
    plotAsLineInc(combined,['Our approach with detection','Incremental','Our approach without detection'],'Test set: Incremental Learning and the effect of unseen classes',fileNamePostfix='_0775Test',std=combinedStd)
    #"""
    
    #"""
    foo = loadCSV('plots/thesisReady/data/IncNovelGWR775-775Dev.csv')
    print('Ours')
    processed, processedStd = transformToMeanStd(foo['F1'],3)
    plotAsBarGWR(processed,[i for i in range(1,12)]*1,r'Increment',None,'',fileNamePostfix='775_Incremental learning with unseen classes',std=processedStd,numberPoints=11)
    print('Novelty')
    processed, processedStd = transformToMeanStd(foo['NovF1'],3)
    plotAsBarGWR(processed,[i for i in range(1,11)]*1,r'Increment',None,'',fileNamePostfix='775_Detection of novel classes',std=processedStd,numberPoints=10)
    #processed, processedStd = transformToMeanStd(foo['IncF1'],3)
    #plotAsBarGWR(processed,[i for i in range(1,12)]*1,r'Increment',None,'Incremental learning',std=processedStd,numberPoints=11)
    foo = loadCSV('plots/thesisReady/data/IncNovelGWR775-775DevNoReject.csv')
    print('Non-reject')
    processed, processedStd = transformToMeanStd(foo['F1'],3)
    plotAsBarGWR(processed,[i for i in range(1,12)]*1,r'Increment',None,'',fileNamePostfix='775_Incremental learning with unseen classes: No detection',std=processedStd,numberPoints=11)
    #processed, processedStd = transformToMeanStd(foo['NovF1'],3)
    #plotAsBarGWR(processed,[i for i in range(11)]*1,r'Increment',None,'GWR Ins 0_825 beta 0_8 F1 Novel',std=processedStd,numberPoints=11)
    print('Incremental')
    processed, processedStd = transformToMeanStd(foo['IncF1'],3)
    plotAsBarGWR(processed,[i for i in range(1,12)]*1,r'Increment',None,'',fileNamePostfix='775_Incremental learning',std=processedStd,numberPoints=11)
    foo = loadCSV('plots/thesisReady/data/IncNovelGWR775-775Test.csv')
    print('Test Ours')
    processed, processedStd = transformToMeanStd(foo['F1'],3)
    plotAsBarGWR(processed,[i for i in range(1,12)]*1,r'Increment',None,'',fileNamePostfix='775_Test set: Incremental learning with unseen classes',std=processedStd,numberPoints=11)
    print('Novelty')
    processed, processedStd = transformToMeanStd(foo['NovF1'],3)
    plotAsBarGWR(processed,[i for i in range(1,11)]*1,r'Increment',None,'',fileNamePostfix='775_Test set: Detection of novel classes',std=processedStd,numberPoints=10)
    #processed, processedStd = transformToMeanStd(foo['IncF1'],3)
    #plotAsBarGWR(processed,[i for i in range(1,12)]*1,r'Increment',None,'Test set: Incremental learning',std=processedStd,numberPoints=11)
    foo = loadCSV('plots/thesisReady/data/IncNovelGWR775-775TestNoReject.csv')
    print('Non-reject')
    processed, processedStd = transformToMeanStd(foo['F1'],3)
    plotAsBarGWR(processed,[i for i in range(1,12)]*1,r'Increment',None,'',fileNamePostfix='775_Test set: Incremental learning with unseen classes: No detection',std=processedStd,numberPoints=11)
    #processed, processedStd = transformToMeanStd(foo['NovF1'],3)
    #plotAsBarGWR(processed,[i for i in range(11)]*1,r'Increment',None,'GWR Ins 0_825 beta 0_8 F1 Novel',std=processedStd,numberPoints=11)
    print('Incremental')
    processed, processedStd = transformToMeanStd(foo['IncF1'],3)
    plotAsBarGWR(processed,[i for i in range(1,12)]*1,r'Increment',None,'',fileNamePostfix='775_Test set: Incremental learning',std=processedStd,numberPoints=11)
    #"""  
    
    
    #"""
    print('Pure GWR With insertion 85 and threshold 825')
    withReject = loadCSV('plots/thesisReady/data/IncNovelPureGWR85-825Dev.csv')    
    #noReject = loadCSV('plots/thesisReady/data/IncNovelGWR775-775DevNoReject.csv')
    method, methodStd = transformToMeanStd(withReject['F1'][0::2],3)
    inc, incStd = transformToMeanStd(withReject['IncF1'][1::2],3)
    fail, failStd = transformToMeanStd(withReject['F1'][1::2],3)
    combined=[method,inc,fail]
    combinedStd=[methodStd,incStd,failStd]
    plotAsLineInc(combined,['Our approach with detection','Incremental','Our approach without detection'],'',fileNamePostfix='Incremental Learning and the effect of unseen classes_PureGWR',std=combinedStd)
    
    withReject = loadCSV('plots/thesisReady/data/IncNovelPureGWR85-825Test.csv')    
    #noReject = loadCSV('plots/thesisReady/data/IncNovelGWR775-775TestNoReject.csv')
    method, methodStd = transformToMeanStd(withReject['F1'][0::2],3)
    inc, incStd = transformToMeanStd(withReject['IncF1'][1::2],3)
    fail, failStd = transformToMeanStd(withReject['F1'][1::2],3)
    combined=[method,inc,fail]
    combinedStd=[methodStd,incStd,failStd]
    plotAsLineInc(combined,['Our approach with detection','Incremental','Our approach without detection'],'',fileNamePostfix='Test set: Incremental Learning and the effect of unseen classes_PureGWR',std=combinedStd)
    
    
    #"""
    foo = loadCSV('plots/thesisReady/data/IncNovelPureGWR85-825Dev.csv')
    print('Ours')
    processed, processedStd = transformToMeanStd(foo['F1'][0::2],3)
    plotAsBarGWR(processed,[i for i in range(1,12)]*1,r'Increment',None,'',fileNamePostfix='PureGWRIncremental learning with unseen classes',std=processedStd,numberPoints=11)
    print('Novelty')
    processed, processedStd = transformToMeanStd(foo['NovF1'][0::2],3)
    plotAsBarGWR(processed,[i for i in range(1,11)]*1,r'Increment',None,'',fileNamePostfix='PureGWRDetection of novel classes',std=processedStd,numberPoints=10)
    #processed, processedStd = transformToMeanStd(foo['IncF1'],3)
    #plotAsBarGWR(processed,[i for i in range(1,12)]*1,r'Increment',None,'Incremental learning',std=processedStd,numberPoints=11)
    #foo = loadCSV('plots/thesisReady/data/IncNovelGWR825-8DevNoReject.csv')
    print('Non-reject')
    processed, processedStd = transformToMeanStd(foo['F1'][1::2],3)
    plotAsBarGWR(processed,[i for i in range(1,12)]*1,r'Increment',None,'',fileNamePostfix='PureGWRIncremental learning with unseen classes: No detection',std=processedStd,numberPoints=11)
    #processed, processedStd = transformToMeanStd(foo['NovF1'],3)
    #plotAsBarGWR(processed,[i for i in range(11)]*1,r'Increment',None,'GWR Ins 0_825 beta 0_8 F1 Novel',std=processedStd,numberPoints=11)
    print('Incremental')
    processed, processedStd = transformToMeanStd(foo['IncF1'][1::2],3)
    plotAsBarGWR(processed,[i for i in range(1,12)]*1,r'Increment',None,'',fileNamePostfix='PureGWRIncremental learning',std=processedStd,numberPoints=11)
    foo = loadCSV('plots/thesisReady/data/IncNovelPureGWR85-825Test.csv')
    print('Test Ours')
    processed, processedStd = transformToMeanStd(foo['F1'][0::2],3)
    plotAsBarGWR(processed,[i for i in range(1,12)]*1,r'Increment',None,'',fileNamePostfix='PureGWRTest set: Incremental learning with unseen classes',std=processedStd,numberPoints=11)
    print('Novelty')
    processed, processedStd = transformToMeanStd(foo['NovF1'][0::2],3)
    plotAsBarGWR(processed,[i for i in range(1,11)]*1,r'Increment',None,'',fileNamePostfix='PureGWRTest set: Detection of novel classes',std=processedStd,numberPoints=10)
    #processed, processedStd = transformToMeanStd(foo['IncF1'],3)
    #plotAsBarGWR(processed,[i for i in range(1,12)]*1,r'Increment',None,'Test set: Incremental learning',std=processedStd,numberPoints=11)
    #foo = loadCSV('plots/thesisReady/data/IncNovelGWR825-8TestNoReject.csv')
    print('Non-reject')
    processed, processedStd = transformToMeanStd(foo['F1'][1::2],3)
    plotAsBarGWR(processed,[i for i in range(1,12)]*1,r'Increment',None,'',fileNamePostfix='PureGWRTest set: Incremental learning with unseen classes: No detection',std=processedStd,numberPoints=11)
    #processed, processedStd = transformToMeanStd(foo['NovF1'],3)
    #plotAsBarGWR(processed,[i for i in range(11)]*1,r'Increment',None,'GWR Ins 0_825 beta 0_8 F1 Novel',std=processedStd,numberPoints=11)
    print('Incremental')
    processed, processedStd = transformToMeanStd(foo['IncF1'][1::2],3)
    plotAsBarGWR(processed,[i for i in range(1,12)]*1,r'Increment',None,'',fileNamePostfix='PureGWRTest set: Incremental learning',std=processedStd,numberPoints=11)
    #"""
    #"""
    
    foo = loadCSV('plots/thesisReady/data/MainResultsCollection.csv')
    selected = foo['F1'][1::2]
    plotAsHorizontalBar(selected,['[Hayashi et al., 2017]','[Choi et al., 2016]','Our approach \n hybrid network','GWR Alone','Challenge baseline \n [Benetos et al., 2016]'],r'Approach',None,'',fileNamePostfix='BatchLearningResults',numberPoints=6)