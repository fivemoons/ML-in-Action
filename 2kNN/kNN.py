# -*- coding: UTF-8 -*-
'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

@author: pbharrin
'''
from numpy import * #导入科学计算包
import operator #运算符模块，排序操作
from os import listdir

def classify0(inX, dataSet, labels, k): #inX 用于分类的输入向量，dataSet训练集，labels标签向量，k选择最近邻居的数目
    dataSetSize = dataSet.shape[0] #训练样本数
    diffMat = tile(inX, (dataSetSize,1)) - dataSet #tile 复制inX dataSetSize行，1列。相当于使inX与训练集每个元素做差值，得到一个矩阵
    sqDiffMat = diffMat**2 #得到差值的平方，得到一个矩阵
    sqDistances = sqDiffMat.sum(axis=1) #按照行对每个差值的平方求和，得到一个向量
    distances = sqDistances**0.5 #开根号，得到一个向量
    sortedDistIndicies = distances.argsort() #对向量进行排序,返回按照排位的元素的下标，[第一小下标，第二小下标，第三小下标]
    classCount={} #类别计数字典
    for i in range(k): #依次寻找每一个近邻
        voteIlabel = labels[sortedDistIndicies[i]] #依次获得最近几个类别
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 #获得类别+1，如果不存在 返回0
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True) #对获得到的类别的多少进行排序
    return sortedClassCount[0][0] #获得最多的类别，[0] 返回第一条记录 [0][0] 返回第一条记录中的key 就是类别

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def file2matrix(filename): #约会，将文件转为数组
    fr = open(filename) #打开文件
    numberOfLines = len(fr.readlines())         #先读一遍，获得稳健的行数
    returnMat = zeros((numberOfLines,3))        #构造空矩阵，3个特征
    classLabelVector = []                       #构造空类别list 
    fr = open(filename)                         #重新打开，文件指针放在开头
    index = 0 #label下标
    for line in fr.readlines():  #读完每一行，转成list
        line = line.strip()  #去掉左右空格
        listFromLine = line.split('\t') #以\t为分隔得到list
        returnMat[index,:] = listFromLine[0:3] #0,1,2放到矩阵中
        classLabelVector.append(int(listFromLine[-1])) #最后一行是类别，追加一个数
        index += 1
    return returnMat,classLabelVector
    
def autoNorm(dataSet):
    minVals = dataSet.min(0) #每一列的最小值
    maxVals = dataSet.max(0) #每一列的最大值
    ranges = maxVals - minVals #每一列的范围
    normDataSet = zeros(shape(dataSet)) #产生一个空矩阵
    m = dataSet.shape[0] #获得矩阵的行数
    normDataSet = dataSet - tile(minVals, (m,1)) #减去最小值
    normDataSet = normDataSet/tile(ranges, (m,1))   #除以尺度，具体每个值相除
    return normDataSet, ranges, minVals
   
def datingClassTest():
    hoRatio = 0.10      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #读入数据
    normMat, ranges, minVals = autoNorm(datingDataMat) #归一化数据
    m = normMat.shape[0] #获得数据实例数
    numTestVecs = int(m*hoRatio) #获得训练集开始位置
    errorCount = 0.0 #错误计数
    for i in range(numTestVecs): #依次使用测试集
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3) #分类i 训练test:m 标签test:m k=3
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0 #增加错误数
    print "the total error rate is: %f" % (errorCount/float(numTestVecs)) #错误率
    print errorCount
    
def img2vector(filename): #把2维图像转为一维向量
    returnVect = zeros((1,1024)) #生成1024维空向量
    fr = open(filename) #读入文件
    for i in range(32): #每一行
        lineStr = fr.readline() #读入一行
        for j in range(32): #每一列
            returnVect[0,32*i+j] = int(lineStr[j]) #放到指定位置
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #把目录中的文件返回成list
    m = len(trainingFileList) #文件个数
    trainingMat = zeros((m,1024)) #训练集
    for i in range(m): #依次读取每一个文件
        fileNameStr = trainingFileList[i] #依次拿出文件名
        fileStr = fileNameStr.split('.')[0]     #去掉后缀名
        classNumStr = int(fileStr.split('_')[0]) #实例的类别
        hwLabels.append(classNumStr) #追加实例的类别
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr) #把指定文件转成1维向量
    testFileList = listdir('testDigits')        #获得测试集目录
    errorCount = 0.0 #错误率
    mTest = len(testFileList) #测试集大小
    for i in range(mTest): #遍历测试集
        fileNameStr = testFileList[i] #得到测试文件名
        fileStr = fileNameStr.split('.')[0]     #去掉后缀名
        classNumStr = int(fileStr.split('_')[0]) #获得实例的类别
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr) #获得测试实例的类别向量
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3) #获得分类结果
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0 #错误计数
    print "\nthe total number of errors is: %d" % errorCount 
    print "\nthe total error rate is: %f" % (errorCount/float(mTest)) #错误率