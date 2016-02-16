# -*- coding: UTF-8 -*-
'''
Created on Oct 27, 2010
Logistic Regression Working Module
@author: Peter
'''
from numpy import *

def loadDataSet(): #便利函数，载入数据
    dataMat = []; labelMat = [] #数据矩阵 标签矩阵
    fr = open('testSet.txt') #打开一个文件
    for line in fr.readlines(): #读入每一行 readlines 一次读入全部
        lineArr = line.strip().split() #去掉左右空格，以tab分隔字符串
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) #加入x0 并且转换成float 并排成list
        labelMat.append(int(lineArr[2])) #加入标签矩阵
    return dataMat,labelMat

def sigmoid(inX): #定义sigmoid函数
    return 1.0/(1+exp(-inX)) #1/(1+e^-z)

def gradAscent(dataMatIn, classLabels): #梯度上升算法
    dataMatrix = mat(dataMatIn)             #转换成Numpy矩阵
    labelMat = mat(classLabels).transpose() #转换成Numpy矩阵
    m,n = shape(dataMatrix) #shape是numpy的函数 返回一个tuple，行 列
    alpha = 0.001 #α参数
    maxCycles = 500 #最大迭代次数
    weights = ones((n,1)) #生成n*1的ndarray ones是numpy函数
    for k in range(maxCycles):              #range是python的函数，返回一个从0开始的list
        h = sigmoid(dataMatrix*weights)     #矩阵乘法 m*n n*1 经过sigmoid函数 得到m*1
        error = (labelMat - h)              #列向量减法 计算误差 m*1
        weights = weights + alpha * dataMatrix.transpose()* error #α乘以的是一个列向量 m*n m*1返回n*1
    return weights

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix) #m样本数 n特征数
    alpha = 0.01
    weights = ones(n)   #权重初始化为行向量 1*n
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights)) #当前行 1*n n*1 求和 sigmoid函数 返回标量
        error = classLabels[i] - h #计算当前误差 返回标量
        weights = weights + alpha * error * dataMatrix[i]  #α乘以一个行向量 1*n
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):#自定义迭代次数 有两点改进
    m,n = shape(dataMatrix)
    weights = ones(n)   #m样本数 n特征数
    for j in range(numIter): #迭代次数
        dataIndex = range(m) #[0,1,2,3...m]
        for i in range(m): #依次循环各个样本
            alpha = 4/(1.0+j+i)+0.0001    #随着迭代次数和样本数的增加而减少
            randIndex = int(random.uniform(0,len(dataIndex)))#生成一个dataIndex长的下标
            h = sigmoid(sum(dataMatrix[randIndex]*weights)) #使用该样本计算预测结果
            error = classLabels[randIndex] - h #计算分类误差
            weights = weights + alpha * error * dataMatrix[randIndex] #更新参数
            del(dataIndex[randIndex]) #从dataIndex这个list中删除该样本
    return weights

def classifyVector(inX, weights): #分类函数，输入数据集和逻辑回归参数
    prob = sigmoid(sum(inX*weights)) #计算该样本的值
    if prob > 0.5: return 1.0 #判为1
    else: return 0.0 #判为0

def colicTest():
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt') #分别读入训练集和测试集
    trainingSet = []; trainingLabels = [] #训练特征 训练标签
    for line in frTrain.readlines():
        currLine = line.strip().split('\t') #用tab分隔
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i])) #一行样本的特征
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000) #使用随机梯度下降1000次
    errorCount = 0; numTestVec = 0.0 #测试分类错误计数 测试样本数
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec) #计算错误率
    print "the error rate of this test is: %f" % errorRate
    return errorRate

def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests): #调用colicTest()函数10次 计算平均值
        errorSum += colicTest()
    print "after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests))
        