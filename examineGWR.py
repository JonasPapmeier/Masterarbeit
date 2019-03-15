# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 21:53:24 2018

@author: brummli
"""

import GWR
import networkx as nx
from networkx.algorithms import community
from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt
import code
import itertools

def transformGWRtoNXGraph(gwr):
    nxGWR = nx.Graph()
    for number, node in enumerate(gwr.nodes):
        label = 0
        nxGWR.add_node(number,label=label)
    for edge in gwr.edges:
        nxGWR.add_edge(edge.nodeFrom,edge.nodeTo)
    return nxGWR
    
def plotGraph(graph,filename):
    fig = plt.figure()
    nx.draw(graph)
    plt.savefig(filename)
    plt.close(fig)
    
def numericLabelToNodeLabels(gwr,labelArray):
    #This function will overwrite existing label and firing information in the gwr
    assert labelArray.shape[0] == len(gwr.nodes)
    maxLabel = np.max(labelArray)
    for num, node in enumerate(gwr.nodes):
        node.label = np.zeros(maxLabel+1)
        node.label[labelArray[num]] = 1
        node.fireCount = 1
        
def edgesToNeighbourMatrix(gwr):
    #For agglomerative clustering with connectivity
    neighMatrix = np.eye(len(gwr.nodes))
    for edge in gwr.edges:
        neighMatrix[edge.nodeFrom,edge.nodeTo] = 1
        #The edges are undirected, the inverse edge should exist in the gwr model
        #but for safety add it here anyway
        neighMatrix[edge.nodeTo,edge.nodeFrom] = 1
    return neighMatrix
    
def girvanNewmanWrapper(gwr,minNumCommunities):
    commGen = community.girvan_newman(gwr)
    commLevels = itertools.takewhile(lambda c: len(c) <= minNumCommunities, commGen)
    for comm in commLevels:
        lastComm = community
    return lastComm
        
def computeCommunities(gwr,function):
    graphNxGWR = transformGWRtoNXGraph(gwr)
    communityLabels = np.zeros(len(gwr.nodes),dtype=np.int32)
    communityGen = function(graphNxGWR)
    for num, comm in enumerate(communityGen):
        for node in comm:
            communityLabels[node] = num
    print('Number communities is {}'.format(np.max(communityLabels)))
    return communityLabels

def weightsArrayFromGWR(gwr):
    weightsArray = np.zeros((len(gwr.nodes),gwr.nodes[0].weights.shape[0]))
    for ind, node in enumerate(gwr.nodes):
        weightsArray[ind] = node.weights
    return weightsArray

def computeClustering(gwr,clusterObject):
    points = weightsArrayFromGWR(gwr)
    clusterLabels = clusterObject.fit_predict(points)
    return clusterLabels

if __name__ == '__main__':
    modelsFolder = 'models/'
    #modelName = modelsFolder+'FIN1-GWR'
    modelName = modelsFolder+'BestGWR_it0.775'
    numCluster = 16
    gwr = GWR.readNetwork(modelName)
    #graph = transformGWRtoNXGraph(gwr)
    #plotGraph(graph,'plots/gwr/BestGWR_it0.775Graph.png')
    #perplexityValues = [10]
    #for perp in perplexityValues:
    #    gwr.plotWithLabel('plots/beamer/gwr/BestGWR_it0775_perplexity'+str(perp)+'.pdf',dimReductionMethod='TSNE',perplexity=perp)
    #commFunction = community.label_propagation.label_propagation_communities
    #commLabels = computeCommunities(gwr,commFunction)
    #numericLabelToNodeLabels(gwr,commLabels)
    #gwr.plotWithLabel('plots/beamer/gwr/test1LabelPropagation'+str(numCluster)+'.png',dimReductionMethod='TSNE')
    #commFunction = lambda x: community.k_clique_communities(x,numCluster)
    #commLabels = computeCommunities(gwr,commFunction)
    #numericLabelToNodeLabels(gwr,commLabels)
    #gwr.plotWithLabel('plots/beamer/gwr/test1kClique'+str(numCluster)+'.png',dimReductionMethod='TSNE')
    #commFunction = lambda x: girvanNewmanWrapper(x,numCluster)
    #commLabels = computeCommunities(gwr,commFunction)
    #numericLabelToNodeLabels(gwr,commLabels)
    #gwr.plotWithLabel('plots/beamer/gwr/test1Centrality'+str(numCluster)+'.png',dimReductionMethod='PCA')
    kmeansClustering = cluster.KMeans(n_clusters=numCluster)
    kmeansLabels = computeClustering(gwr,kmeansClustering)
    numericLabelToNodeLabels(gwr,kmeansLabels)
    gwr.plotWithLabel('plots/beamer/gwr/BestGWR_it0775_test1KMeans'+str(numCluster)+'.pdf',dimReductionMethod='TSNE',perplexity=10)
    connectivity = edgesToNeighbourMatrix(gwr)
    aggloClustering = cluster.AgglomerativeClustering(n_clusters=numCluster,connectivity=connectivity)
    aggloLabels = computeClustering(gwr,aggloClustering)
    numericLabelToNodeLabels(gwr,aggloLabels)
    gwr.plotWithLabel('plots/beamer/gwr/BestGWR_it0775_test1allgo'+str(numCluster)+'.pdf',dimReductionMethod='TSNE',perplexity=10)
    #code.interact(local=locals())