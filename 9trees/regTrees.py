# -*- coding: utf-8 -*-
'''
Created on Feb 4, 2011
Tree-Based Regression Methods
@author: Peter Harrington
'''
from numpy import *

def loadDataSet(fileName):      #从文件中读入一个dataMat
    dataMat = []                #假设最后一列是标签
    fr = open(fileName)         #打开文件
    for line in fr.readlines(): #readlines()读入全部行
        curLine = line.strip().split('\t') #strip()去掉左右空格 split()以 \t分隔 返回list
        fltLine = map(float,curLine) #map函数，将float函数依次作用到curLine list的每一个元素中
        dataMat.append(fltLine) #追加，新生成一行
    return dataMat #返回数据集 最后一列是标签

def binSplitDataSet(dataSet, feature, value): #二分数据集 数据集合，待切分的特征，该特征的某个值
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:][0] #返回了feature的值大于value的行
    # feature=1 dataSet=[[x,x,x],[0,1,1],[x,x,x]]
    #dataSet[:,feature] 数据集的第feature列的全部，[[0],[1],[1]] 每个list只有一个元素
    #其中行>value的值 转换成是否 [[False],[True],[True]]
    #nonzero()返回了对应的元素的下标矩阵,由于前一行是二维所以返回也是两个二维，[[[1,2]] [[0,0]]] 即（1,2）（2,0）是大于0的
    #nonzero()[0]选取了对应的行信息 [[1,2]] 说明第二行和第三行是满足条件的行
    #dataSet[nonzero()[0],:]取得这些满足条件的行的全部内容，[[[x,1,x][x,1,x]]]
    #dataSet[nonzero()[0],:][0] 去掉一层[] 得到[[x,1,x],[x,1,x]]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:][0] #返回了feature的值小于等于value的行
    return mat0,mat1 #返回按照dataSet分隔好的两个子集合

def regLeaf(dataSet):#目标变量的均值，负责生成叶节点，该数据集不能细分时计算
    return mean(dataSet[:,-1]) #拿出最后一列，计算均值

def regErr(dataSet): #计算目标变量的平方误差
    return var(dataSet[:,-1]) * shape(dataSet)[0] #var均方差*样本的总数

def linearSolve(dataSet):   #将数据格式化成X Y
    m,n = shape(dataSet) #返回dataSet的大小
    X = mat(ones((m,n))); Y = mat(ones((m,1))) #构建X的大小和Y的大小
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1] #X第一列是1，第二列到最后是实例的特征，Y得到实例的回归值
    xTx = X.T*X #xTx是 X的转置*X
    if linalg.det(xTx) == 0.0:  #判断X是否是奇异矩阵，如果是，不可逆
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y) #得到线性回归系数
    return ws,X,Y #返回回归系数和X，Y

def modelLeaf(dataSet):#当数据不再需要切分的时候，负责生成回归模型，返回回归系数
    ws,X,Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet): #在给定数据集上计算误差
    ws,X,Y = linearSolve(dataSet) #返回数据集的回归系数
    yHat = X * ws #预测回归值
    return sum(power(Y - yHat,2)) #求给定数据集上的平方误差

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)): #选择最好的特征进行切分，如果找不到好的切分，返回None和产生叶子节点
    tolS = ops[0]; tolN = ops[1] #tolS容许的误差下降值，tolN切分的最少样本数
    #退出条件1：如果所有剩余标签的值只有一个，则退出，返回None和将数据集放到同一个叶子节点里计算均值
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #拿出最后一列，转置，变成list，去掉一层方括号，转成set，计算长度
        return None, leafType(dataSet)
    m,n = shape(dataSet) #计算当前数据集的大小，m个实例，n-1个特征
    #选择最好特征的方法：RSS误差的减小 平均误差的总值，总方差
    S = errType(dataSet) #计算当前数据集的RSS误差
    bestS = inf; bestIndex = 0; bestValue = 0 #初始化当前的最好误差为正无穷，最好下标为0，最好的切分值为0
    for featIndex in range(n-1): #依次枚举每一个特征
        for splitVal in set(dataSet[:,featIndex]): #依次枚举每一个特征中的取值
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal) #将数据集二分成两部分，按照每个特征的取值都切分一遍
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue #如果切分后的两个数据集中有一个小于了要求的最小样本数，放弃
            newS = errType(mat0) + errType(mat1) #分别计算两个数据集的RSS误差，作为当前切分的RSS误差
            if newS < bestS: #如果当前误差比best要好
                bestIndex = featIndex #更新最好的特征下标
                bestValue = splitVal #更新最好的特征切分值
                bestS = newS #g更新最小的RSS误差
    #退出条件2：如果当前数据集无论怎么切分都不能让RSS的减少满足阈值，说明当前的数据集已经足够集中，返回None和数据集作为叶子节点的均值
    if (S - bestS) < tolS: 
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue) #按照当前最好的二分方式来切分数据集
    #退出条件3：疑问：这个条件一定不满足，因为循环中满足这个条件的都continue了，只能是初始值inf，0,0才可能满足这个条件，但是inf时条件2已经return了
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex,bestValue #返回最好的特征下标和最好的切分值

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):#数据集NumPy mat，建立叶节点的函数，误差计算函数，树构建的参数元组
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)#选择最好的划分
    if feat == None: return val #如果当前数据集不可切分，回归树则返回数据集的常数值，模型树则返回数据集的线性方程
    retTree = {} #可二分，创建一个树，用字典表示
    retTree['spInd'] = feat #该节点的切分特征的下标
    retTree['spVal'] = val #该节点的切分特征的值
    lSet, rSet = binSplitDataSet(dataSet, feat, val) #切分后的左子树和右子树的数据集
    retTree['left'] = createTree(lSet, leafType, errType, ops) #递归构造左子树
    retTree['right'] = createTree(rSet, leafType, errType, ops) #递归构造右子树
    return retTree #返回构造的字典树


#以下代码是后剪纸的代码
def isTree(obj): #判断一个变量是不是树
    return (type(obj).__name__=='dict') #

def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right']) #如果右结点是棵树，递归计算右子树，放到右子树的值中
    if isTree(tree['left']): tree['left'] = getMean(tree['left']) #同理
    return (tree['left']+tree['right'])/2.0 #计算左右子树的回归值的平均值
    
def prune(tree, testData): #剪枝函数，tree要剪纸的树，testData测试集
    if shape(testData)[0] == 0: return getMean(tree) #如果没有测试集，则返回全部剪枝
    if (isTree(tree['right']) or isTree(tree['left'])): #如果如果左右至少有一棵是子树
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal']) #则按照原决策树的样子把测试集拆分成两部分
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet) #如果左子树是树，则递归调用剪枝函数
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet) #如果右子树是树，则递归调用剪枝函数
    #而如果左右两个都不是树，则判断是不是需要合并他们两个结点
    if not isTree(tree['left']) and not isTree(tree['right']): #两个都不是树
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal']) #切分数据集分到左右
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) +\
            sum(power(rSet[:,-1] - tree['right'],2))    #sum（（测试集的回归值-左子树的回归值）^2）合并前的误差
        treeMean = (tree['left']+tree['right'])/2.0 #合并后的回归值
        errorMerge = sum(power(testData[:,-1] - treeMean,2)) #合并后的误差
        if errorMerge < errorNoMerge: #如果合并之后误差更小
            print "merging" #打印合并
            return treeMean #返回合并后的回归值
        else: return tree #没有合并，返回原树
    else: return tree #疑问：两个至少有一个树和都不是树都已经讨论过了，程序不会运行到这里
    
def regTreeEval(model, inDat): #回归树的计算
    return float(model)

def modelTreeEval(model, inDat):#模型树的计算
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model)

def treeForeCast(tree, inData, modelEval=regTreeEval):#testForeCast自顶向下遍历整棵树，直到命中叶节点为止。一旦到达叶节点则调用modelEval
    #tree 是已经构建好的决策树
    #inData 需要预测的输入数据
    #modelEval是对叶节点数据进行预测的函数的引用
    if not isTree(tree): return modelEval(tree, inData) #如果是叶子节点，则调用Eval函数计算回归或者模型预测值
    if inData[tree['spInd']] > tree['spVal']: #判断当前树的第一个特征，走分支
        if isTree(tree['left']): return treeForeCast(tree['left'], inData, modelEval) #如果左分支是树，递归，带入左分支的树
        else: return modelEval(tree['left'], inData) #如果左分支是节点，计算其预测值
    else: #如果是右分支
        if isTree(tree['right']): return treeForeCast(tree['right'], inData, modelEval) #如果右分支是树，递归，带入右分支的树
        else: return modelEval(tree['right'], inData) #如果右分支是节点，计算其预测值
        
def createForeCast(tree, testData, modelEval=regTreeEval): #tree是已经训练好的树 testData是测试集 modelEval选择回归还是模型
    m=len(testData) #测试集大小
    yHat = mat(zeros((m,1))) #获得一个m行1列的空矩阵
    for i in range(m): #枚举每一行
        yHat[i,0] = treeForeCast(tree, mat(testData[i]), modelEval) #推算每一个测试数据
    return yHat #返回预测出来的回归值

#使用 corrcoef(yHat, testMat[:,1],rowvar = 0)[0,1] 返回协方差矩阵