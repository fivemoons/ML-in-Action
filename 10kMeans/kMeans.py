# -*- coding: utf-8 -*-
'''
Created on Feb 16, 2011
k Means Clustering for Ch10 of Machine Learning in Action
@author: Peter Harrington
'''
from numpy import *

def loadDataSet(fileName):      #读入数据
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine) #把curLine中的每一个元素都转换成float格式
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA, vecB): #计算欧式距离的函数
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)    求平方后再求和 然后开方

def randCent(dataSet, k): #初始构建k个质心
    n = shape(dataSet)[1] #有多少列
    centroids = mat(zeros((k,n)))#生成的中心矩阵 k行n列 k：质心数 n：特征数
    for j in range(n):#生成随机质心，在数据的最小最大范围内
        minJ = min(dataSet[:,j])  #计算每一种特征的最小值
        rangeJ = float(max(dataSet[:,j]) - minJ) #最大值和最小值的范围
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1)) #范围中随机
    return centroids #返回随机出来的质心
    
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent): #数据集 质心数据 计算距离函数 创建随机质心函数
    m = shape(dataSet)[0] #样本数
    clusterAssment = mat(zeros((m,2)))#标记每一个样本分到了哪个质心
                                      #2：记录簇索引值，记录和质心之间的误差
    centroids = createCent(dataSet, k) #使用创建质心的函数初始化质心矩阵
    clusterChanged = True #质心矩阵是否改变，跳出flag
    while clusterChanged: #当改变了
        clusterChanged = False #设为为改变
        for i in range(m):#计算每一个质心是否需要改变
            minDist = inf; minIndex = -1 #最近的距离 最近的下标
            for j in range(k): #依次看每一个质心
                distJI = distMeas(centroids[j,:],dataSet[i,:]) #计算质心和该数据点的距离
                if distJI < minDist: #如果可以更新
                    minDist = distJI; minIndex = j #更新下距离和下标 注意：质心下标可能没有更换
            if clusterAssment[i,0] != minIndex: clusterChanged = True #确保只有更换的时候才将flag设为True
            clusterAssment[i,:] = minIndex,minDist**2 #更新该组数据点的质心，误差信息
        print centroids #打印当前循环后的质心变动情况
        for cent in range(k):#重新计算质心的位置
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]] #将质心是当前点的簇坐标拿出来 nonzero返回一个tuple
            centroids[cent,:] = mean(ptsInClust, axis=0) #这个行向量 等于 ptsInClust的行向量
    return centroids, clusterAssment

def biKmeans(dataSet, k, distMeas=distEclud): #初始是一个大点，从一个大点继续往后分
    m = shape(dataSet)[0] #样本数
    clusterAssment = mat(zeros((m,2))) #存放分类
    centroid0 = mean(dataSet, axis=0).tolist()[0] #计算所有样本中心的数据
    centList =[centroid0] #放到质心列表中
    for j in range(m):#依次计算每一个样本的质心误差
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2 #计算这个节点的误差
    while (len(centList) < k): #当误差小于k时
        lowestSSE = inf #最小误差
        for i in range(len(centList)): #依次遍历每一个质心
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]#将质心是当前点的簇坐标拿出来
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas) #计算这些点分成2分时的质心数据 和 数据所在的质心
            sseSplit = sum(splitClustAss[:,1])#计算二分部分的sse
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1]) #计算没有二分的sse
            print "sseSplit, and notSplit: ",sseSplit,sseNotSplit
            if (sseSplit + sseNotSplit) < lowestSSE: #比较分割后是否是最优
                bestCentToSplit = i #分割点
                bestNewCents = centroidMat #最优质心分隔结果
                bestClustAss = splitClustAss.copy() #最优样本误差
                lowestSSE = sseSplit + sseNotSplit #最优sse
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #更新簇的更新结果，没有二分的地方
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit #更新簇的更新结果，二分的地方
        print 'the bestCentToSplit is: ',bestCentToSplit
        print 'the len of bestClustAss is: ', len(bestClustAss)
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0] #将二分列表更新一个
        centList.append(bestNewCents[1,:].tolist()[0]) #再追加一个
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#重新计算各点所在质心
    return mat(centList), clusterAssment

import urllib
import json
def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'  #create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J'#JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.urlencode(params)
    yahooApi = apiStem + url_params      #print url_params
    print yahooApi
    c=urllib.urlopen(yahooApi)
    return json.loads(c.read())

from time import sleep
def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print "%s\t%f\t%f" % (lineArr[0], lat, lng)
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else: print "error fetching"
        sleep(1)
    fw.close()
    
def distSLC(vecA, vecB):#Spherical Law of Cosines
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * \
                      cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b)*6371.0 #pi is imported with numpy

import matplotlib
import matplotlib.pyplot as plt
def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()
