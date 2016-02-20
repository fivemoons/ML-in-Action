# -*- coding: utf-8 -*-
'''
Created on Jun 1, 2011

@author: Peter Harrington
'''
from numpy import *

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName) #打开文件
    stringArr = [line.strip().split(delim) for line in fr.readlines()] #读入切割生成列表[[]]
    datArr = [map(float,line) for line in stringArr] #每一个行的每一个元素转换成float
    return mat(datArr) #转换成矩阵

def pca(dataMat, topNfeat=9999999): #dataMat数据集[[..]..]
    meanVals = mean(dataMat, axis=0) #首先计算平均数，按照列计算平均数 [[..]]
    meanRemoved = dataMat - meanVals #减去平均值
    covMat = cov(meanRemoved, rowvar=0) #计算协方差
    eigVals,eigVects = linalg.eig(mat(covMat)) #计算协方差矩阵的特征值 eigVals：特征值 eigVects：特征向量
    eigValInd = argsort(eigVals)            #对特征值进行排序
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #只保留最大的几个特征向量
    redEigVects = eigVects[:,eigValInd]       
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals #把数据转换回低维数据
    return lowDDataMat, reconMat

def replaceNanWithMean(): 
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i]) #values that are not NaN (a number)
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal  #set NaN values to mean
    return datMat
