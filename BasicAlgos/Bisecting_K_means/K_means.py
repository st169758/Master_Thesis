import numpy as np
import pandas as pd
from numpy import *
import warnings
warnings.filterwarnings("ignore")

#load the filename of a txt data file 
#and output a list contains all the data
def loadDataSet(filename):
    dataMat=[]
    fr=open(filename)
    
    for line in fr.readlines():
        curLine=line.strip().split('\t')
        fltLine=list(map(float,curLine))
        dataMat.append(fltLine)
    
    return dataMat

#Calculate the Euclidean distance between any two vectors
def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))

#Create a set of random K centroids for each feature of the given dataset
def randCent(dataSet,K):
    n=shape(dataSet)[1]
    centroids=mat(zeros((K,n)))
    for j in range(n):
        minValue=min([dataSet[i][j] for i in range(len(dataSet))])
        maxValue=max([dataSet[i][j] for i in range(len(dataSet))])
        rangeValues=float(maxValue-minValue)
        #Make sure centroids stay within the range of data
        centroids[:,j]=minValue+rangeValues*random.rand(K,1)
    return centroids

#Implementation of K means clustering method
#The last two parameters in the function can be omitted.
#Output the matrix of all centroids and a matrix (clusterAssment) whose first column represents the 
#belongings of clusters of each obvservation and second column represents the SSE for each
#observation
def kMeans(dataSet,K,distMethods=distEclud,createCent=randCent):
    m=shape(dataSet)[0]
    clusterAssment=mat(zeros((m,2)))
    centroids=createCent(dataSet,K)
    clusterChanged=True

    while clusterChanged:
        clusterChanged=False
        for i in range(m):
            minDist=inf; minIndex=-2
            for j in range(K):
                distJI=distMethods(centroids[j,:],dataSet[i])
                if distJI<minDist:
                    minDist=distJI;minIndex=j
            if clusterAssment[i,0] != minIndex:
                clusterChanged=True
            clusterAssment[i,:]=minIndex,minDist**2

        dataSet = np.array(dataSet)
        #update all the centroids by taking the mean value of relevant data
        for cent in range(K):
            ptsInClust=dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]
            centroids[cent,:]=mean(ptsInClust,axis=0)
    return centroids,clusterAssment           


#bisecting K-means method
def bisectingKMeans(dataSet,K,numIterations=1):
    m,n = shape(dataSet)
    clusterInformation=mat(zeros((m,2)))
    centroidList=[]
    minSSE=inf
    
    #At the first place, regard the whole dataset as a cluster and find the best clusters
    for i in range(numIterations):
        centroid,clusterAssment=kMeans(dataSet, 2)
        SSE=sum(clusterAssment,axis=0)[0,1]
        if SSE<minSSE:
            minSSE=SSE
            tempCentroid=centroid
            tempCluster=clusterAssment
    centroidList.append(tempCentroid[0].tolist()[0])
    centroidList.append(tempCentroid[1].tolist()[0])
    clusterInformation=tempCluster
    minSSE=inf 
    
    while len(centroidList)<K:
        maxIndex=-2
        maxSSE=-1
        #Choose the cluster with Maximum SSE to split
        for j in range(len(centroidList)):
            SSE=sum(clusterInformation[nonzero(clusterInformation[:,0]==j)[0]])
            if SSE>maxSSE:
                maxIndex=j
                maxSSE=SSE
                
        minIndex=-2
        dataSet = np.array(dataSet)
        #Choose the clusters with minimum total SSE to store into the centroidList
        for k in range(numIterations):
            pointsInCluster=dataSet[nonzero(clusterInformation[:,0]==maxIndex)[0]]
            centroid,clusterAssment=kMeans(pointsInCluster, 2)
            SSE=sum(clusterAssment[:,1],axis=0)

            if SSE<minSSE:
                minSSE=SSE
                tempCentroid=centroid.copy()
                tempCluster=clusterAssment.copy()

        #Update the index
        tempCluster[nonzero(tempCluster[:,0]==1)[0],0]=len(centroidList)
        tempCluster[nonzero(tempCluster[:,0]==0)[0],0]=maxIndex
        print(tempCluster.shape)
        #update the information of index and SSE

        clusterInformation[nonzero(clusterInformation[:,0]==maxIndex)[0],:]=tempCluster
        #update the centrolist
        centroidList[maxIndex]=tempCentroid[0].tolist()[0]
        centroidList.append(tempCentroid[1].tolist()[0])

    y_Bisecting_kmeans = []
    for each in dataSet:
        dist_list = []
        for k in range(K):
            dist_list.append(distEclud(each,centroidList[k]))
        # print(len(dist_list))
        y_Bisecting_kmeans.append(np.argmin(dist_list))

    return y_Bisecting_kmeans
        
if __name__ == '__main__':
    data = loadDataSet('./testSet.txt')
    y_bis_kmeans = bisectingKMeans(dataSet=data, K=5)
    print(y_bis_kmeans)
