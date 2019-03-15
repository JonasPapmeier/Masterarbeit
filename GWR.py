import numpy
#import theano


from numpy import eye, asarray, dot, sum 
from numpy.linalg import svd
from functools import partial

import matplotlib as mpl
import latexFormatPlotter
mpl.use('Agg')
#mpl.rcParams.update(latexFormatPlotter.nice_fonts)
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap,TSNE
from sklearn.preprocessing import StandardScaler

# Growing When Required Network.
# Code from Marsland 2003, https://seat.massey.ac.nz/personal/s.r.marsland/gwr.html
# Ported by Pablo Barros, 2016
# Speed optimization by Jonas Papmeier, 2018

def readNetwork(readDirectory): 
    targetFile = open(readDirectory, 'r')
    edges = []
    nodes = []
    fakeData = []
    readingEdge = True
    paramsLine = targetFile.readline()
    paramsInfo = paramsLine.split(',')
    for line in targetFile:    
        if not line == "----\n":
            if readingEdge:
                edgeInformation = line.split(",")
                edge = Edge(nodeFrom=int(edgeInformation[0]), nodeTo=int(edgeInformation[1]), age=int(edgeInformation[2]))
                edges.append(edge)
            else:
                nodeInformation = line.split(";")
                node = Node(habn =float(nodeInformation[0]), distance= float(nodeInformation[1]), fireCount=int(nodeInformation[2]))
                
                label = nodeInformation[3].split(",")
                label = [float(i) for i in label]
                label = numpy.array(label)
                node.label = label
                
                weights = nodeInformation[4].split(",")
                
                weights = [float(i) for i in weights] 
                weights = numpy.array(weights)
                node.weights = weights
                fakeData.append(weights)
                
                neighbours = nodeInformation[5].split(",")
                neighbours = [int(i) for i in neighbours] 
                node.neighbours = neighbours
                nodes.append(node)
                
        else:
            readingEdge = False                
    targetFile.close()    
    
    gwr = GWR(fakeData,max_epochs=int(paramsInfo[0]),hab_threshold=float(paramsInfo[1]),insertion_threshold=float(paramsInfo[2]),
              epsilon_b=float(paramsInfo[3]),epsilon_n=float(paramsInfo[4]),tau_b=float(paramsInfo[5]),tau_n=float(paramsInfo[6]),
              MAX_NODES=int(paramsInfo[7]),MAX_NEIGHBOURS=int(paramsInfo[8]),MAX_AGE=int(paramsInfo[9]))
    gwr.nodes = nodes
    gwr.edges = edges
    gwr.updateWeightList()
    return gwr

class Edge:
    def __init__(self, nodeFrom=None, nodeTo=None, age=0):
        self.nodeFrom = nodeFrom
        self.nodeTo = nodeTo
        self.age = age
        
class Node:
    
    def __init__(self, habn=0,distance=0,old=0,fireCount=0,neighbour=0,label=None,weights=[], neighbours=[]):
        self.habn = habn
        self.fireCount = fireCount
        self.distance = distance
        self.old = old
        self.weights = weights
        self.neighbours = neighbours
        self.label=label
        
#TODO:remove
def calculateEuclideanDistanceFromPair(pointPair):
    point1 = pointPair[0]
    point2 = pointPair[1]
    return numpy.exp(-numpy.sum(numpy.square(point1-point2)))

def calculateEuclideanDistance(point2,node):
   point1 = node.weights
   return numpy.exp(-numpy.sum(numpy.square(point1-point2)))

class GWR:    
#Input functions
#% The Grow When Required Network
#% Stephen Marsland, 2000 - 2003
#% stephen.marsland@man.ac.uk
#% Inputs:   
#%   datafile - specified data - no_of_datapoints by no_of_dimensions (blank or '' causes a file dialog to appear)
#%   weightfile - weightfile to load from (blank means none, '' causes a file dialog to appear)
#%   outfile - file to save the network (blank or '' means that one is
#%   generated automatically)
#%   do_plot - 1: plot (2D) figures, 0: don't
#% Parameters to be set (specified in the code):
#%   max_epochs, insert_threshold, epsilon_b, epsilon_n, tau_b, tau_n, hab_threshold,
#%   MAX_NODES, MAX_EDGES, MAX_NEIGHBOURS, MAX_AGE
#%   The learning rule (lines 84,91) and node position rule (line 368) -
#%   comment out the one you don't want to use
#% The figure plotting only plots the first 2 dimensions. I should fix this, I
#% know. 


    #Network Parameters
#    max_epochs = 50
#    hab_threshold = 0.1
#    insertion_threshold = 0.7
#    epsilon_b = 0.5
#    epsilon_n = 0.05
#    tau_b = 0.3
#    tau_n = 0.1
#    MAX_NODES = 30000
#    MAX_EDGES = 3*MAX_NODES
#    MAX_NEIGHBOURS = 10
#    MAX_AGE = 100

    max_epochs = 50
    hab_threshold = 0.3
    insertion_threshold = 0.5
    epsilon_b = 0.5
    epsilon_n = 0.05
    tau_b = 0.3
    tau_n = 0.1
    MAX_NODES = 2000
    MAX_EDGES = 3*MAX_NODES
    MAX_NEIGHBOURS = 3
    MAX_AGE = 2
    
    def __init__(self, data, max_epochs=50, hab_threshold=0.3, insertion_threshold=0.5, epsilon_b=0.5, epsilon_n=0.05, tau_b=0.3, tau_n=0.1, MAX_NODES = 2000, MAX_NEIGHBOURS = 10, MAX_AGE = 3):
        
                
        self.max_epochs = max_epochs
        self.hab_threshold = hab_threshold
        self.insertion_threshold = insertion_threshold
        self.epsilon_b = epsilon_b
        self.epsilon_n = epsilon_n
        self.tau_b = tau_b
        self.tau_n = tau_n
        self.MAX_NODES = MAX_NODES
        self.MAX_EDGES = 3*MAX_NODES
        self.MAX_NEIGHBOURS = MAX_NEIGHBOURS
        self.MAX_AGE = MAX_AGE
        
        #To compute mean and std easily online
        self.scaler = StandardScaler()
        self.data = data
        #assume pre-shuffled data
        self.numberInputs = numpy.shape(data)[1]
        self.initializeNetwork(data)
        print('Node weights shape', self.nodes[0].weights.shape)
        
        self.weightsList = [self.nodes[0].weights,self.nodes[1].weights]
        self.weightsChanged = True

    #Save the Network
    def save(self, saveDirectory):         
        targetFile = open(saveDirectory, 'w')
        
        targetFile.write(str(self.max_epochs)+','+str(self.hab_threshold)+','+
        str(self.insertion_threshold)+','+str(self.epsilon_b)+','+str(self.epsilon_n)+
        ','+str(self.tau_b)+','+str(self.tau_n)+','+str(self.MAX_NODES)+','
        +str(self.MAX_NEIGHBOURS)+','+str(self.MAX_AGE)+'\n')        
        for edge in self.edges:  
                        
            targetFile.write(str(edge.nodeFrom))
            targetFile.write(",")
            targetFile.write(str(edge.nodeTo))
            targetFile.write(",")                
            targetFile.write(str(edge.age))
            targetFile.write("\n")                     
        targetFile.write("----")                   
        targetFile.write("\n")  
        for node in self.nodes:            
            targetFile.write(str(node.habn))
            targetFile.write(";")    
            targetFile.write(str(node.distance))
            targetFile.write(";")                    
            targetFile.write(str(node.fireCount))  
            targetFile.write(";")
            for labelIndex in range(len(node.label)):
                targetFile.write(str(node.label[labelIndex]))
                if not labelIndex == len(node.label)-1:
                    targetFile.write(",")
            targetFile.write(";")
            for weight in range(len(node.weights)):
                targetFile.write(str(node.weights[weight]))
                if not weight == len(node.weights)-1:
                    targetFile.write(",")
            targetFile.write(";")                
            for neihbour in range(len(node.neighbours)):
                targetFile.write(str(node.neighbours[neihbour]))
                if not neihbour == len(node.neighbours)-1:
                    targetFile.write(",")
            targetFile.write("\n")                                
            
        targetFile.close()
        
    """
    Method defaults to PCA
    """
    def reduceWeightsDimensions(self,weightsArray,dim,method=None,perplexity=10,learning_rate=10,n_iter=5000,n_iter_without_progress=300,early_exaggeration=12.0):
        if method == 'Isomap':
            reduceModel = Isomap(n_components=dim)
        elif method == 'TSNE':
            reduceModel = TSNE(n_components=dim,verbose=1,method='barnes_hut',init='random',perplexity=perplexity,learning_rate=learning_rate,n_iter=n_iter,n_iter_without_progress=n_iter_without_progress,early_exaggeration=early_exaggeration,random_state=1337)
        else:
            reduceModel = PCA(n_components=dim)
        reducedWeightsArray = reduceModel.fit_transform(weightsArray)
        return reducedWeightsArray
        
    def plot3DHelper(self,fileName,nodeDataArray,edgesDataArray,labelArray,edgeNodeMapArray):
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        ax.view_init(7, -80)
        colorMap = mpl.cm.get_cmap(name='Set1')
        maxLabel = numpy.max(labelArray)
        colorsArray = colorMap(labelArray/maxLabel,alpha=1)
        classes = range(int(maxLabel)+1)
        class_colours = list(map(colorMap,map(lambda x: x/(maxLabel*1.0),classes)))
        recs = []
        for i in range(0,len(class_colours)):
            recs.append(mpl.patches.Rectangle((0,0),1,1,fc=class_colours[i]))        
        ax.scatter(nodeDataArray[:,0], nodeDataArray[:,1], nodeDataArray[:,2],c=colorsArray)
        plt.legend(recs,classes,loc='best').get_frame().set_alpha(0.5)
        plt.savefig(fileName)
        for i in range(0,edgesDataArray.shape[0],2):
            ax.plot(edgesDataArray[i:i+2,0],edgesDataArray[i:i+2,1],edgesDataArray[i:i+2,2],c=colorsArray[int(edgeNodeMapArray[i])])
        plt.savefig(fileName[:-4]+'_withEdges.png')
        plt.close(fig)
        
    def plot2DHelper(self,fileName,nodeDataArray,edgesDataArray,labelArray,edgeNodeMapArray):
        fig, ax = plt.subplots(1, 1, figsize=latexFormatPlotter.set_size(latexFormatPlotter.textwidthBeamer))
        colorMap = mpl.cm.get_cmap(name='Paired')
        maxLabel = numpy.max(labelArray)
        colorsArray = colorMap(labelArray/maxLabel,alpha=1)
        classes = range(int(maxLabel)+1)
        classNames = ['background','clearthroat','cough','doorslam', 'drawer', 'keyboard', 'keys', 'knock', 'laughter', 'pageturn', 'phone', 'speech']
        class_colours = list(map(colorMap,map(lambda x: x/(maxLabel*1.0),classes)))
        recs = []
        for i in range(0,len(class_colours)):
            recs.append(mpl.patches.Rectangle((0,0),1,1,fc=class_colours[i]))
        plt.scatter(nodeDataArray[:,0], nodeDataArray[:,1],c=colorsArray,edgecolors='k',s=20)
        plt.legend(recs,classes,loc='best').get_frame().set_alpha(0.5)
        ax.tick_params(axis='both',bottom=False,left=False,labelbottom=False,labelleft=False)
        plt.savefig(fileName, format='pdf', bbox_inches='tight')
        for i in range(0,edgesDataArray.shape[0],2):
            plt.plot(edgesDataArray[i:i+2,0],edgesDataArray[i:i+2,1],c=colorsArray[int(edgeNodeMapArray[i])])
        plt.savefig(fileName[:-4]+'_withEdges.pdf', format='pdf', bbox_inches='tight')
        plt.close(fig)
        
    def plotWithLabel(self,fileName,dim=2,dimReductionMethod=None,perplexity=10,learning_rate=50,n_iter=5000,n_iter_without_progress=300,early_exaggeration=12.0):
        try:
            maxLabel = 0
            labelsNormalizedArray = numpy.zeros(len(self.nodes))
            for num, node in enumerate(self.nodes):
                labelArray = numpy.copy(node.label)
                fireCount = node.fireCount# + 1
                labelArray = labelArray/fireCount
                threshold = 0.7
                labelArray = numpy.where(labelArray>=threshold,1,0)
                label = numpy.argmax(numpy.concatenate((numpy.zeros(1),labelArray),axis=-1),axis=-1)
                labelsNormalizedArray[num] = label
                maxLabel = max(maxLabel,label)
            weightsArray = numpy.array(self.weightsList)
            if weightsArray.shape[-1] > 3:
                weightsArray = self.reduceWeightsDimensions(weightsArray,dim,method=dimReductionMethod,perplexity=perplexity,learning_rate=learning_rate,n_iter=n_iter,n_iter_without_progress=n_iter_without_progress,early_exaggeration=early_exaggeration)
            edgePoints = numpy.zeros((len(self.edges)*2,weightsArray.shape[-1]))
            edgeNodes = numpy.zeros((len(self.edges)*2,1))
            for num, edge in enumerate(self.edges):
                edgePoints[num*2,:] = weightsArray[edge.nodeFrom]
                edgePoints[num*2+1,:] = weightsArray[edge.nodeTo]
                edgeNodes[num*2] = edge.nodeFrom
                edgeNodes[num*2+1] = edge.nodeTo
            if dim ==2:
                self.plot2DHelper(fileName,weightsArray,edgePoints,labelsNormalizedArray,edgeNodes)
            else:
                self.plot3DHelper(fileName,weightsArray,edgePoints,labelsNormalizedArray,edgeNodes)
        except AttributeError:
            print('Some nodes are missing a label. Please run a labelling algorithm first')
            
    def plotWithConnectedComponents(self,fileName,dim=2,dimReductionMethod=None,perplexity=30):
        try:
            for node in self.nodes:
                node.component
        except AttributeError:
            self.findConnectedComponents()
        weightsArray = numpy.array(self.weightsList)
        if weightsArray.shape[-1] > 3:
            weightsArray = self.reduceWeightsDimensions(weightsArray,dim,method=dimReductionMethod,perplexity=perplexity)
        labelsNormalizedArray = numpy.zeros(weightsArray.shape[0])
        for num, node in enumerate(self.nodes):
            labelsNormalizedArray[num] = node.component
        edgePoints = numpy.zeros((len(self.edges)*2,weightsArray.shape[-1]))
        edgeNodes = numpy.zeros((len(self.edges)*2,1))
        for num, edge in enumerate(self.edges):
            edgePoints[num*2,:] = weightsArray[edge.nodeFrom]
            edgePoints[num*2+1,:] = weightsArray[edge.nodeTo]
            edgeNodes[num*2] = edge.nodeFrom
            edgeNodes[num*2+1] = edge.nodeTo
        if dim ==2:
            self.plot2DHelper(fileName,weightsArray,edgePoints,labelsNormalizedArray,edgeNodes)
        else:
            self.plot3DHelper(fileName,weightsArray,edgePoints,labelsNormalizedArray,edgeNodes)
        
        
    #Initialize the network
    # Create two nodes which matches two random positions in the input data
    def initializeNetwork(self, data):
        self.nodes = []        
        for i in range(2):
            randomDataPoint = self.data[i]
            node = Node(habn=1,distance=0,old=0,neighbour=[], weights = randomDataPoint, neighbours=[])        
            self.nodes.append(node)
        self.edges = []
    
    #TODO: clean up findWinners function
    def findWinnersParallel(self,dataPoint,workerPool):
        #nodePointPairs = list(map(lambda node: (node.weights, dataPoint), self.nodes))
        #nodeDistances = list(workerPool.map(calculateEuclideanDistanceFromPair, nodePointPairs))
        
        distanceFunc = partial(calculateEuclideanDistance, dataPoint)
        nodeDistances = list(workerPool.map(distanceFunc, self.nodes,chunksize=5000))
        
        indexBest = numpy.argmax(nodeDistances)
        distanceBest = nodeDistances[indexBest]
        nodeDistances[indexBest] = 0
        indexSecondBest = numpy.argmax(nodeDistances)
        return indexBest,indexSecondBest, distanceBest
        
    def findWinnersAssumeUpdatedWeightList(self,dataPoint):
        diff = dataPoint - self.weightsList
        squared = numpy.square(diff)
        summed = -numpy.sum(squared,axis=1)
        nodeDistances=summed
        
        indexBest = numpy.argmax(nodeDistances)
        #Calculate only activation of neuron before returning
        #Distance has to be only sum to avoid all distances becoming zero
        distanceBest = numpy.exp(nodeDistances[indexBest])
        #Remove best matching from list to find second best
        nodeDistances[indexBest] = numpy.iinfo(numpy.int32).min  #-1 is enough as distance function is bounded between 0 and 1
        indexSecondBest = numpy.argmax(nodeDistances)
        
        return indexBest,indexSecondBest, distanceBest
    
    def updateWeightList(self):
        newWeightsList = []
        for index, node in enumerate(self.nodes):
            newWeightsList.append(node.weights)
        self.weightsList = newWeightsList
        self.weightsChanged = False
    
    #Find the two closest BMU from a dataPoint    
    def findWinners(self, dataPoint,computeActivation=False):
        
        if self.weightsChanged:
            self.updateWeightList()
        diff = self.weightsList - dataPoint
        squared = numpy.square(diff)
        summed = -numpy.sum(squared,axis=1)
        nodeDistances = summed
        
        #Doesn't set distance for all nodes
        #If really needed somewhere the iteration version needs to be used
        #but that version is slower
        indexBest = numpy.argmax(nodeDistances)
        if computeActivation:
            distanceBest = numpy.exp(nodeDistances[indexBest])
        else:
            distanceBest = nodeDistances[indexBest]
        nodeDistances[indexBest] = numpy.iinfo(numpy.int32).min
        indexSecondBest = numpy.argmax(nodeDistances)
        
       
        return indexBest,indexSecondBest, distanceBest
        
    def getNeighboursOrderedDistance(self, bmu, dataPoint):
        
        node = self.nodes[bmu]
        returnList = []
        
        for n in node.neighbours:
            
            self.nodes[int(n)].distance = numpy.exp(-numpy.sum((self.nodes[int(n)].weights-dataPoint)**2))
            returnList.append(self.nodes[int(n)])
            
        newList = sorted(returnList, key=lambda x: x.distance, reverse=True)
        returnList = []
        for i in range(len(newList)):
            returnList.append(self.nodes.index(newList[i]))
        
        return returnList
        
    # Delete a node(indexNode2) from the neighbours list of another node(indexNode1)
    def deleteNeighbour(self, indexNode1, indexNode2):
        try:
            self.nodes[indexNode1].neighbours.remove(indexNode2)
        except:
            print("Index1:", indexNode1)
            print("Index2:", indexNode2)
            print("Nodes:", len(self.nodes))
                
    # Made two nodes neighbours of each other
    def addNeighbours(self,indexNode1, indexNode2):
        if len(self.nodes[indexNode1].neighbours) < self.MAX_NEIGHBOURS:
            self.nodes[indexNode1].neighbours.append(indexNode2)
        else:
            print('Node neighbour rejected:', indexNode1, indexNode2)
            
        if len(self.nodes[indexNode2].neighbours) < self.MAX_NEIGHBOURS:
            self.nodes[indexNode2].neighbours.append(indexNode1)
        else:
            print('Node neighbour rejected2:', indexNode1, indexNode2)            


    #Check whether an edge exists between 2 nodes, return its number of -1 if 
    # it does not exist    
    def findEdge(self,indexNode1,indexNode2):
        
        #Search for the edges and return the first one
        if not self.edges == []:
            for index, edge in enumerate(self.edges):
                if (edge.nodeFrom == indexNode1 and edge.nodeTo == indexNode2) or (edge.nodeFrom == indexNode2 and edge.nodeTo == indexNode1):
                    return index
        #In case of not finding an edge, return -1            
        return -1
    
    #Age all edges with an end at the best node
    def ageEdge(self, bmuIndex):
        
        removeEdges = []
        #Update the age of all edges connected to the BMU
        updatedEdges = 0
        for index, edge in enumerate(self.edges):
            if edge.nodeFrom == bmuIndex or edge.nodeTo == bmuIndex:
                edge.age = edge.age+1
                updatedEdges = updatedEdges + 1
                if edge.age > self.MAX_AGE:
                    removeEdges.append(index)
                #Break early when all edges to BMU have been updated
                if updatedEdges >= len(self.nodes[bmuIndex].neighbours):
                    break

        removeNodes = [] #Remove nodes that have no neighbours
        for removeIndex in reversed(removeEdges):
            fromNode = self.edges[removeIndex].nodeFrom
            toNode = self.edges[removeIndex].nodeTo
            self.deleteEdge(edgeNumber=removeIndex)
            if len(self.nodes[fromNode].neighbours) < 1:
                removeNodes.append(fromNode)
            if len(self.nodes[toNode].neighbours) < 1:
                removeNodes.append(toNode)
        for removeIndex in sorted(removeNodes,reverse=True):
            #Remove nodes immediately, should be correct according to paper,
            #is implemented different in original author implementation
            self.deleteNode(removeIndex)
            pass


    #Delete Edges.
    # Specify the two nodes or
    # specify the index of the edge
    def deleteEdge(self, indexNode1=None, indexNode2=None, edgeNumber=-1):
        
        if edgeNumber == -1:             
            edgeNumber = self.findEdge(indexNode1, indexNode2)
            assert edgeNumber >= 0
        
        if edgeNumber >= 0:
            if edgeNumber >= len(self.edges) or edgeNumber<0:
                print("Edges:", len(self.edges))
                print("Edge Number:", edgeNumber)
                print("indexNode1:", indexNode1)
                print("indexNode2:", indexNode2)               
            self.deleteNeighbour(self.edges[edgeNumber].nodeFrom, self.edges[edgeNumber].nodeTo)
            self.deleteNeighbour(self.edges[edgeNumber].nodeTo, self.edges[edgeNumber].nodeFrom)
            
            self.edges.pop(edgeNumber)
        

     #Add an edge between the two nodes
    # or set the age to 0 if it already exists
    def addEdge(self, indexNode1,indexNode2):
        
        newEdge = 0
        
        
        #Add an edge between the nodes, if possible
        if (indexNode2 not in self.nodes[indexNode1].neighbours) and (indexNode1 not in self.nodes[indexNode2].neighbours):
            if len(self.edges) < self.MAX_EDGES:
                edge = Edge(nodeFrom =indexNode1, nodeTo = indexNode2, age = 0)
                self.edges.append(edge)
                self.addNeighbours(indexNode1,indexNode2)
                newEdge = 1
            else:
                print("Number maximum of edges reached!")
                print("Number of edges:", len(self.edges))
        else:
            #Find the existing edge
            edgeIndex = self.findEdge(indexNode1,indexNode2)
            assert edgeIndex != -1
            self.edges[edgeIndex].age = 0
            newEdge = 2
        return newEdge     
        

    # Insert a new node
    def insertNode(self, indexNode1, indexNode2, dataPoint):
       
       if len(self.nodes) < self.MAX_NODES:
           self.weightsChanged = True
           node = Node(habn=1,distance=0,old=0,neighbour=[], weights = dataPoint, neighbours=[])        
           node.weights = self.nodes[indexNode1].weights  - (( self.nodes[indexNode1].weights - dataPoint) / 2)
           self.nodes.append(node)
           self.weightsList.append(node.weights)
           self.addEdge(indexNode1, len(self.nodes)-1)
           self.addEdge(indexNode2, len(self.nodes)-1)
           self.deleteEdge(indexNode1,indexNode2)
           
    def deleteNode(self, nodeIndex):
      # Update the indeces of the nodes in the nodes neighbour lists       
      # Iterate through all the neighbours and update the index of the node by reducing one position
      # if the node was above (index position) the node which was deleted 
       for indexNode in range(len(self.nodes)):
           
           if nodeIndex in self.nodes[indexNode].neighbours:
               print('Warning deleted node had a neighbour')
               self.deleteNeighbour(indexNode, nodeIndex)
            
           for neighbourIndex in range(len(self.nodes[indexNode].neighbours)):
               if self.nodes[indexNode].neighbours[neighbourIndex] > nodeIndex:
                  self.nodes[indexNode].neighbours[neighbourIndex] -= 1
                                 
       
        #Update the edge from and to list using the same principle as above            
       deleteEdgesList = []
       for indexEdge in range(len(self.edges)):
          if self.edges[indexEdge].nodeFrom == nodeIndex or self.edges[indexEdge].nodeTo == nodeIndex:
              deleteEdgesList.append(indexEdge)
       edgeLen = len(self.edges)
       for removeIndex in reversed(deleteEdgesList):
            self.deleteEdge(edgeNumber=removeIndex)
       assert len(self.edges) == edgeLen-len(deleteEdgesList)
                    
       for indexEdge in range(len(self.edges)):
            if self.edges[indexEdge].nodeFrom > nodeIndex:
              self.edges[indexEdge].nodeFrom = self.edges[indexEdge].nodeFrom - 1
            if self.edges[indexEdge].nodeTo > nodeIndex:
              self.edges[indexEdge].nodeTo = self.edges[indexEdge].nodeTo -1
                
       self.nodes.pop(nodeIndex)
       self.weightsList.pop(nodeIndex)
       self.weightsChanged = True
       
    #Habituate a value       
    def habituate(self, value, tau):
        return tau * 1.05 * (1 - value) - tau;

    def getCodebook(self):
        codebook = []
        for node in self.nodes:
                codebook.append(node.weights)
                
        return numpy.array(codebook)
    
    def depthFirstSearcher(self,nodeIndex,visited, component):
        visited[nodeIndex] = True
        component.append(nodeIndex)
        self.nodes[nodeIndex].component = len(self.connectedComponents)
        for neighboursIndex in self.nodes[nodeIndex].neighbours:
            if visited[neighboursIndex] == False:
                self.depthFirstSearcher(neighboursIndex,visited,component)
    
    #Oriented at https://www.geeksforgeeks.org/depth-first-search-or-dfs-for-a-graph/
    #Last accessed 20.04.2018, 01:08
    def findConnectedComponentsOld(self):
        visited = [False for i in range(len(self.nodes))]
        self.connectedComponents = []
        
        for nodeIndex in range(len(self.nodes)):
            if visited[nodeIndex] == False:
                component = []
                self.depthFirstSearcher(nodeIndex, visited, component)
                self.connectedComponents.append(component)
    
    #Rewritten to iterative instead of recursive, because max recursive stack size can be hit otherwise
    def findConnectedComponents(self):
        visited = [False for i in range(len(self.nodes))]
        self.connectedComponents = []
        
        for nodeIndexGlobal in range(len(self.nodes)):
            if not visited[nodeIndexGlobal]:
                component = []
                nodeNeighbourStack = []
                nodeNeighbourStack.append(nodeIndexGlobal)
                while len(nodeNeighbourStack) > 0:
                    nodeIndex = nodeNeighbourStack.pop()
                    if not visited[nodeIndex]:
                        visited[nodeIndex] = True
                        component.append(nodeIndex)
                        self.nodes[nodeIndex].component = len(self.connectedComponents)
                        nodeNeighbourStack.extend(self.nodes[nodeIndex].neighbours)
                self.connectedComponents.append(component)


    def varimax(self, Phi, gamma = 1.0, q = 20, tol = 1e-6):
        p,k = Phi.shape
        R = eye(k)
        d=0
        for i in xrange(q):
            d_old = d
            Lambda = dot(Phi, R)
            u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, numpy.diag(numpy.diag(dot(Lambda.T,Lambda))))))
            R = dot(u,vh)
            d = sum(s)
            if d_old!=0 and d/d_old < 1 + tol: break
        return dot(Phi, R)
                        
    #Train a GWR
    def train(self, data=None, labels=None, epochs=0):
        if epochs > 0:
            maxEpochs = epochs
        else:
            maxEpochs = self.max_epochs
        currentEpoch = 0

        if not data is None:
            self.data=data
            
        if labels is None:
            labels = [numpy.zeros(1) for i in range(len(self.data))]
        else:
            print(labels.shape)

        while currentEpoch < maxEpochs:
            currentEpoch = currentEpoch +1             
            newNeurons = 0
            updatedNeurons = 0
            newEdges = 0
            updatedEdges = 0
            neuronsRemovedInThisEpoch=[]
            dataCount = 0
            for dataIndex in range(len(self.data)):
                dataPoint = self.data[dataIndex]
                label =  numpy.copy(labels[dataIndex])
                #Find the two best-matching nodes
                bmu1,bmu2, bmu1Distance = self.findWinnersAssumeUpdatedWeightList(dataPoint)
                self.scaler.partial_fit(bmu1Distance)
                #Add an edge between the nodes
                newEdge = self.addEdge(bmu1,bmu2)
                if newEdge == 1:
                    newEdges = newEdges+1
                elif newEdge ==2:
                    updatedEdges = updatedEdges+1
                if self.nodes[bmu1].habn < self.hab_threshold and bmu1Distance < self.insertion_threshold:
                    self.insertNode(bmu1,bmu2,dataPoint)
                    self.nodes[-1].label = label
                    self.nodes[-1].fireCount += 1
                    newNeurons = newNeurons+1
                else:
                    updatedNeurons = updatedNeurons+1
                    #Train the best node already there - two choices of training rule
                    self.nodes[bmu1].weights = self.nodes[bmu1].weights + self.epsilon_b*self.nodes[bmu1].habn * (dataPoint-self.nodes[bmu1].weights)
                    
                    self.weightsList[bmu1] = self.nodes[bmu1].weights 
                    self.nodes[bmu1].habn = self.nodes[bmu1].habn + self.habituate(self.nodes[bmu1].habn, self.tau_b)
                    self.nodes[bmu1].fireCount += 1
                    #Accumulate the occurenc of each label in the node
                    if self.nodes[bmu1].label is not None:
                        self.nodes[bmu1].label = self.nodes[bmu1].label + label
                    else:
                        self.nodes[bmu1].label = label
                    
                    #Train the neighbours
                    for neighbourIndex in range(len(self.nodes[bmu1].neighbours)):
                        
                        self.nodes[neighbourIndex].weights = self.nodes[neighbourIndex].weights + self.epsilon_n*self.nodes[neighbourIndex].habn * (dataPoint-self.nodes[neighbourIndex].weights)
                        self.weightsList[neighbourIndex] = self.nodes[neighbourIndex].weights
                        self.nodes[neighbourIndex].habn = self.nodes[neighbourIndex].habn + self.habituate(self.nodes[neighbourIndex].habn, self.tau_n)
                    self.ageEdge(bmu1)
                    self.weightsChanged = True
                
                dataCount +=1
                if dataCount % 10000 == 0:
                    print('At iteration {} with currently {} nodes'.format(dataCount,len(self.nodes)))
                 
            nodesToBeRemoved = []
            for indexNodes in range(len(self.nodes)):
                if len(self.nodes[indexNodes].neighbours) == 0:
                    nodesToBeRemoved.append(indexNodes)                        
                    neuronsRemovedInThisEpoch.append(indexNodes)     
                    
            for deleteNodes in range(len(nodesToBeRemoved)):
                self.deleteNode(nodesToBeRemoved[deleteNodes])
                for ai in range(len(nodesToBeRemoved)):
                    nodesToBeRemoved[ai] = nodesToBeRemoved[ai]-1
            
            print("---------------------")
            print("DataPoints:", len(self.data))
            print("Epoch:", currentEpoch)
            print("newNeurons:", newNeurons)
            print("updatedNeurons:", updatedNeurons)
            print("newEdges:", newEdges)
            print("updatedEdges:", updatedEdges)
            print("nodesToBeRemoved:", len(neuronsRemovedInThisEpoch))
            print("Total Neurons:", len(self.nodes))
            print("Total Edges:", len(self.edges))
            print("Activation mean {} and var {}".format(self.scaler.mean_,self.scaler.var_))
            print("---------------------")
            newNeurons = 0
            neuronsRemovedInThisEpoch = []

if __name__ == '__main__':
    #Some simple test code to get a feeling for the way a GWR works and trains    
    
    import code
    from sklearn.datasets.samples_generator import make_swiss_roll
    from sklearn.datasets import make_blobs, load_digits
    n_samples = 1500
    noise = 0.05
    X, Y = make_blobs(n_samples=n_samples,n_features=100,centers=10)
    print(Y.shape)
    y_categorical = numpy.zeros((Y.shape[0],numpy.max(Y)+1))
    for i in range(y_categorical.shape[0]):
        y_categorical[i,Y[i]]=1
    print(y_categorical)
    Y=y_categorical
    #X, _ = load_digits(n_class=10,return_X_y=True)
    #X, _ = make_swiss_roll(n_samples, noise)
    #X = numpy.arange(300)
    #X = numpy.reshape(X,(100,3))
    #X = X*2
    #numpy.random.shuffle(X)
    # Make it thinner
    #X[:, 1] *= 1.5
    X = (X-numpy.min(X,axis=0,keepdims=True)) / (numpy.max(X,axis=0,keepdims=True) - numpy.min(X,axis=0,keepdims=True) + 1e-7)*2-1
    X = X / numpy.sqrt(numpy.mean(numpy.square(X),axis=-1,keepdims=True))
    X = X*1
    fig = plt.figure()
    #ax = p3.Axes3D(fig)
    #ax.view_init(7, -80)
    plt.scatter(X[:,0], X[:,1])
    plt.savefig('plots/gwr/test.png')
    plt.close(fig)
    gwr = GWR(X,max_epochs=20,hab_threshold=0.6,insertion_threshold=0.05,epsilon_b=0.5,epsilon_n=0.000001,tau_b=0.3,tau_n=0.1,MAX_NODES=2000,MAX_NEIGHBOURS=100,MAX_AGE=6)
    gwr.train(labels=Y)
    gwr.plotWithLabel('plots/gwr/testLabel.pdf')
    code.interact(local=locals())
