# -*- coding: UTF-8 -*-
'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers'] #不浮出水面是否能够生存 是否有脚蹼
    #change to discrete values
    return dataSet, labels

def calcShannonEnt(dataSet):#计算给定数据集的香农熵
    numEntries = len(dataSet)  #数据集中的实例总数
    labelCounts = {} #声明一个字典，用来记录每个标签的实例数
    for featVec in dataSet: #依次遍历数据集中的每一行实例
        currentLabel = featVec[-1] #获得当前实例的最后一列，即标签
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0 #如果当前实例的标签不存在标签字典中，引入key
        labelCounts[currentLabel] += 1 #对应的标签+1
    #至此，计算出了所有标签的个数存放在labelCounts字典中
    shannonEnt = 0.0 #默认香农熵为0
    for key in labelCounts: #依次遍历每一种标签
        prob = float(labelCounts[key])/numEntries #当前标签所占的比例
        shannonEnt -= prob * log(prob,2) # p*log(2,p)本身是个负数，所以香农熵要减去这个值
    return shannonEnt
    
def splitDataSet(dataSet, axis, value):#按照给定特征划分数据集，待划分的数据集，划分数据集的特征，需要返回的特征的值
    retDataSet = [] #python是引用传递，所以要新建一个数据集
    for featVec in dataSet: #依次遍历数据集中的每一行实例
        if featVec[axis] == value: #找出对应特征是某个值的实例
            reducedFeatVec = featVec[:axis]     #从0到axis-1的列加入到新的list reducedFeatVes中
            reducedFeatVec.extend(featVec[axis+1:]) #extend是在当前list中扩展 axis+1到末尾的值
            retDataSet.append(reducedFeatVec) #append是追加，是在二维list中再加一行list
    return retDataSet #返回新生成的划分好的数据集
    
def chooseBestFeatureToSplit(dataSet): #选择最好的数据集划分方式
    numFeatures = len(dataSet[0]) - 1      #最后一列是标签，numFeatures是特征的数目
    baseEntropy = calcShannonEnt(dataSet) #保存了整个数据集的原始香农熵，保存最初的无序度量值
    bestInfoGain = 0.0; bestFeature = -1 #bestInfoGain是最好的划分特征对应的香农熵 bestFeature是最好的特征编号
    for i in range(numFeatures):        #依次选择每一个特征来计算，即每一列
        featList = [example[i] for example in dataSet] #列表推导式目的是产生一个list，条件是datsSet中每一行的第i个元素
        uniqueVals = set(featList)       #set目的：获得featList中的唯一值。目的：得到当前特征的取值有哪些
        newEntropy = 0.0 #使用这个特征划分下的新的熵
        for value in uniqueVals: #遍历当前特征中的所有唯一属性值 目的：计算每种划分方式的信息熵
            subDataSet = splitDataSet(dataSet, i, value) #对每个特征的取值划分一次数据集。注意：每个特征的每个取值都要进行划分
            prob = len(subDataSet)/float(len(dataSet)) #包含特定特征的子数据集与整个数据集之比
            newEntropy += prob * calcShannonEnt(subDataSet)    #x1划分下：newEntropy = p(x1=0)*H(x1=0)划分之后的一颗子树的熵 + p(x1=1)*H(x1=1) 划分之后的另外一颗子树的熵
        infoGain = baseEntropy - newEntropy     #原来熵是最高的，不确定性最高，这个特征划分之后，熵变小了，infoGain是熵的增益值。就是这次划分使得熵变小了多少
        if (infoGain > bestInfoGain):       #获得熵增益最大的那个划分
            bestInfoGain = infoGain         #当前最大的熵增益
            bestFeature = i                 #能够获得当前最大的熵增益的特征标号
    return bestFeature                      #返回当前最好的特征

def majorityCnt(classList):   #多数表决函数 classList:类别列表
    classCount={} #标签计数字典
    for vote in classList: #遍历列表中的每一个元素
        if vote not in classCount.keys(): classCount[vote] = 0 #如果不在dict中，则dict增加该标签，并设为0
        classCount[vote] += 1 #该类别标签计数+1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True) #classCount的key值排序，排序内容是value，逆序排序 #返回一个tuple的list
    return sortedClassCount[0][0] #[("yes",200),("no"),100]   返回yes

def createTree(dataSet,labels): #dataSet中包含特征，最后一列是标签，labels是特征的名字
    classList = [example[-1] for example in dataSet] #classLists存放标签，一个list
    if classList.count(classList[0]) == len(classList):  #递归函数的第一个停止条件：所有的标签都是同一个类别的。
        return classList[0] #如果是同一个类别的则返回当前类别，不用再次划分
    if len(dataSet[0]) == 1: #递归函数的第二个停止条件是，所有的特征都是用完了，即dataSet中只剩下标签了
        return majorityCnt(classList) #使用多数表决程序返回当前数据集的分类结果
    bestFeat = chooseBestFeatureToSplit(dataSet) #选择当前最好的特征划分数据集，bestFeat当前最好的特征
    bestFeatLabel = labels[bestFeat] #bestFeatLabel ：当前最好的特征叫什么名字
    myTree = {bestFeatLabel:{}} #当前最好的特征构造一棵字典类型的树，myTree
    del(labels[bestFeat]) #删除labels最好特征的名字，通过下标
    featValues = [example[bestFeat] for example in dataSet] #featValues 列表生成式 各个实例最好特征的取值[1,1,1,1,0,0,0,1,0,0]
    uniqueVals = set(featValues) #转换成set([1,0]) 去掉重复值
    for value in uniqueVals: #依次遍历当前类别的每一个取值
        subLabels = labels[:]       #拷贝一份labels，让下面的递归调用不更改labels的值
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels) #依次生成树的每颗子树，按照特征的不同取值切分数据集
    return myTree
    
def classify(inputTree,featLabels,testVec): #使用决策树的分类函数 构造好的决策树
    firstStr = inputTree.keys()[0] #决策树的根结点
    secondDict = inputTree[firstStr] #获得根结点下的子树，一个key对应一个子树
    featIndex = featLabels.index(firstStr) #获得名字是firstStr的对应的特征编号
    key = testVec[featIndex] #testVec的对应的标号的特征的取值
    valueOfFeat = secondDict[key] #根据取值获得对应的子树
    if isinstance(valueOfFeat, dict):  #如果对应的子树是一个dict，说明还可以继续分类。
        classLabel = classify(valueOfFeat, featLabels, testVec) #递归寻找子树对应的标签，valueOfFeat是一颗子树
    else: classLabel = valueOfFeat #如果不是一个dict 说明已经是一个分类的结果
    return classLabel #返回分类的结果

def storeTree(inputTree,filename): #序列化对象，pickle 是python的模块 inputTree需要序列化的对象,filename需要保存的文件名
    import pickle #引入pickle模块
    fw = open(filename,'w') #写入一个文件
    pickle.dump(inputTree,fw) #pickle.dump函数写入一个对象进入文件
    fw.close() #关闭文件
    
def grabTree(filename): #反序列化对象
    import pickle #引入pickle模块
    fr = open(filename) #读入一个文件
    return pickle.load(fr) #pickle.load函数从文件读入一个对象
    
