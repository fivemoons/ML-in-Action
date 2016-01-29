# -*- coding: UTF-8 -*-
'''
Created on Nov 28, 2010
Adaboost is short for Adaptive Boosting
@author: Peter
'''
from numpy import *

def loadSimpData():  #导入简单数据
    datMat = matrix([[ 1. ,  2.1], #特征坐标
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0] #类别
    return datMat,classLabels

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#训练集，用于划分的特征，特征值阈值，小于还是大于
    retArray = ones((shape(dataMatrix)[0],1)) #全为1的划分数组，小于返回-1 大于返回1
    if threshIneq == 'lt': #如果是小于
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0 #数组过滤
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray
    

def buildStump(dataArr,classLabels,D): #找到数据集上最佳的单层决策树
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T #获得训练集，和类别集合
    m,n = shape(dataMatrix) #m实例数 n特征维度
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1))) #numSteps遍历特征的所有可能值 bestStump最佳决策树的相关信息 
    minError = inf #用于寻找最小错误率
    for i in range(n):#遍历所有特征
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max(); #获得特征中的最小值和最大值
        stepSize = (rangeMax-rangeMin)/numSteps #分10步遍历，每步的大小
        for j in range(-1,int(numSteps)+1):#从-1遍历到11
            for inequal in ['lt', 'gt']: #是小于还是大于
                threshVal = (rangeMin + float(j) * stepSize) #依次枚举每一个切分点
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#使用第i个特征，threshVal的切分点，大于或小于 返回划分数组
                errArr = mat(ones((m,1))) #获得误差列向量，默认为1
                errArr[predictedVals == labelMat] = 0 #分类正确的设为0
                weightedError = D.T*errArr  #权重数组和误差01数组相乘得到总的误差，标量
                #print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError: #更新最小误差
                    minError = weightedError #最小错误率
                    bestClasEst = predictedVals.copy() #保存最优划分
                    bestStump['dim'] = i #特征是i
                    bestStump['thresh'] = threshVal #划分值是threshVal
                    bestStump['ineq'] = inequal #小于还是大于
    return bestStump,minError,bestClasEst


def adaBoostTrainDS(dataArr,classLabels,numIt=40): #输入数据集，类别标签，迭代次数，DS使用单层决策树
    weakClassArr = [] #弱分类器数组
    m = shape(dataArr)[0] #实例数
    D = mat(ones((m,1))/m)   #初始化D列向量为1/m 之和为1
    aggClassEst = mat(zeros((m,1))) #记录每个数据点的类别估计累计值
    for i in range(numIt): #迭代次数
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)#使用权值D建立一个弱分类器
        #print "D:",D.T
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))#根据分类误差计算该弱分类器的比重，并根据此值计算D，max是防止error==0
        bestStump['alpha'] = alpha  #记录当前弱分类器的比重
        weakClassArr.append(bestStump)                  #把该弱分类器的特征，划分值，小于大于，比重保存起来
        #print "classEst: ",classEst.T
        expon = multiply(-1*alpha*mat(classLabels).T,classEst) #指数部分，-比重*类别行向量*分类结果列向量
        D = multiply(D,exp(expon))                      #D是初始权重，计算后D是还没有除规范化因子的新权重
        D = D/D.sum() #规范化因子就是D的求和，让D之和变为1
        #calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha*classEst #类别估计累计值，原来有个估计值，现在加上新的弱分类器，累加
        #print "aggClassEst: ",aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1))) #累计误差等于 两个行向量不等为1乘以1矩阵 得到错误的个数
        errorRate = aggErrors.sum()/m #计算错误率
        print "total error: ",errorRate 
        if errorRate == 0.0: break #如果错误率为0 则退出迭代
    return weakClassArr,aggClassEst

def adaClassify(datToClass,classifierArr): #利用训练出来的多个弱分类器进行分类
    dataMatrix = mat(datToClass)#待分类样本
    m = shape(dataMatrix)[0] #实例数
    aggClassEst = mat(zeros((m,1))) # 构造一个列向量，计算每个实例的估计累计值
    for i in range(len(classifierArr)): #依次拿出弱分类器
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])#调用分类函数，使用当前弱分类器得到分类结果
        aggClassEst += classifierArr[i]['alpha']*classEst #加权后累加进入估计累计值
        print aggClassEst
    return sign(aggClassEst) #根据每一个实例的符号 范湖一个列向量

def plotROC(predStrengths, classLabels): #分类器的预测强度列向量，类别标签
    import matplotlib.pyplot as plt
    cur = (1.0,1.0) #构造一个浮点数二元组，初始位1,1
    ySum = 0.0 #用于计算AUC的值
    numPosClas = sum(array(classLabels)==1.0) #通过数组过滤计算classLabels==1.0的个数 正例的个数
    yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels)-numPosClas) #根据正例个数计算y轴的步长，根据负例个数计算x周的步长
    sortedIndicies = predStrengths.argsort() #根据所有实例的预测强度排序，从最小到最大。返回的是数组的索引值，predeS是列向量所以返回二维array，且(*,0)
    fig = plt.figure() #绘画对象
    fig.clf() #清空当前绘画对象
    ax = plt.subplot(111) #创建一个1行1列的图，ax是第一个图
    #预测强度是个二维array转换成list,（*，0） 中 [0] 取出行号
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0: #如果对应的类别标签是1
            delX = 0; delY = yStep; #沿着y轴方向下降1步，降低真阳率
        else: #如果是-1或0
            delX = xStep; delY = 0; #沿着x轴方向左移一步，降低假阴率
            ySum += cur[1] #所有高度的和ySum只有在沿X轴移动时才会增加
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b') #从起始点到移动点画线
        cur = (cur[0]-delX,cur[1]-delY) #计算新的坐标点
    ax.plot([0,1],[0,1],'b--') #
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print "the Area Under the Curve is: ",ySum*xStep
