# -*- coding: UTF-8 -*-
'''
Created on Oct 19, 2010

@author: Peter
'''
from numpy import *

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1] #1表示侮辱性的言论，0表示不是
    return postingList,classVec  #返回样本 和 标签
                 
def createVocabList(dataSet): #从dataSet中返回词汇列表
    vocabSet = set([])  #创建不重复词列表
    for document in dataSet: #依次枚举dataSet中的每一个列表。
        vocabSet = vocabSet | set(document) #set(document)返回一个不重复的集合，与vocabset并集
    return list(vocabSet) #将set类型转换回list类型

def setOfWords2Vec(vocabList, inputSet): #任意输入一个inpuSet，把出现的单词在vocabList中设为1
    returnVec = [0]*len(vocabList) #创建一个与词汇表等长的list，并初始化为0
    for word in inputSet: #遍历inputSet中的所有单词
        if word in vocabList: #如果单词在vocabList中
            returnVec[vocabList.index(word)] = 1 #单词表中该单词对应的下标设为1
        else: print "the word: %s is not in my Vocabulary!" % word
    return returnVec #返回转换后的单词向量

def trainNB0(trainMatrix,trainCategory): #训练样本，标签
    numTrainDocs = len(trainMatrix) #样本数
    numWords = len(trainMatrix[0]) #每个样本的长度，已经使用setOfWords2Vec转换为向量
    pAbusive = sum(trainCategory)/float(numTrainDocs) #p(c1) 计算正样本的概率
    p0Num = ones(numWords); p1Num = ones(numWords)    #初始化两个全1的向量，防止出现0概率的问题
    p0Denom = 2.0; p1Denom = 2.0                      #更改分母为2，防止出现0概率的问题
    for i in range(numTrainDocs): #依次枚举每一个样本，i是下标
        if trainCategory[i] == 1: #如果样本的标签是1，说明是侮辱性言论
            p1Num += trainMatrix[i] #p1Num是一个向量，累加上当前样本出现的单词
            p1Denom += sum(trainMatrix[i]) #p1Denom是一个实数，计算侮辱性言论的单词的总数
        else:
            p0Num += trainMatrix[i] #非侮辱性言论
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)          #change to log() p(wi|c1) 对于p1Num向量的每一个元素都除以总词数 numpy的性质
    p0Vect = log(p0Num/p0Denom)          #change to log() p(wi|c0) 防止出现过小数 则p(w0|c0)*p(w1|c0) 改成 log(p(w0|c0))+log(p(w1|c0))
    return p0Vect,p1Vect,pAbusive #返回p(wi|c1) p(wi|c0) p(c1)

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1): #要分类的言论向量，上个函数返回的三个概率
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #vec2Classify * p1Vec两个向量逐个元素相乘，得到该言论侮辱词汇向量，然后sum求和
    #即p(w0|c1)*p(w1|c1)*...*p(wn|c1)   在加上log(pClass1) 相当于*p(c1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1) #同上
    if p1 > p0: #最大似然估计 返回概率大的类别
        return 1
    else: 
        return 0
    
def bagOfWords2VecMN(vocabList, inputSet): #词袋模型 每个词可以出现多次
    returnVec = [0]*len(vocabList) #生成空数组
    for word in inputSet: #每一个输入向量
        if word in vocabList: #每一个单词
            returnVec[vocabList.index(word)] += 1 #单词对应的下标+1
    return returnVec

def testingNB(): #遍历函数
    listOPosts,listClasses = loadDataSet() #载入 样本 和 标签
    myVocabList = createVocabList(listOPosts) #计算单词标签
    trainMat=[] #初始化训练集
    for postinDoc in listOPosts: #依次枚举每一个样本
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc)) #计算样本单词转换成向量
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses)) #p(w|c0) p(w|c1) p(c)
    testEntry = ['love', 'my', 'dalmation'] #生成测试样本
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry)) #转换成单词向量
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb) #计算分类结果
    testEntry = ['stupid', 'garbage'] #生成测试样本2
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry)) #转换成单词向量2
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb) #计算分类结果2




#示例：使用朴素贝叶斯过滤垃圾邮件
def textParse(bigString):    #分词函数，输入是一个大字符串，输出时单词list
    import re #引入正则
    listOfTokens = re.split(r'\W*', bigString) #匹配单词
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] #列表生成器 变小写 少于2个字符的忽略
    
def spamTest(): #
    docList=[]; classList = []; fullText =[]
    for i in range(1,26): #对于每一个输入文件
        wordList = textParse(open('email/spam/%d.txt' % i).read())  #读入垃圾邮件列表
        docList.append(wordList) #docList是[]  加入一条样本
        fullText.extend(wordList) #将单词列表全部加入到列表中
        classList.append(1) #类别list增加1 标记为正样本
        wordList = textParse(open('email/ham/%d.txt' % i).read()) #读入正常邮件列表
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0) #类别list增加0 标记为负样本
    vocabList = createVocabList(docList)     #创建单词列表
    trainingSet = range(50); testSet=[]      #训练集0..49 测试集 空
    for i in range(10):                      #随机选择10个数据当做测试集
        randIndex = int(random.uniform(0,len(trainingSet)) #随机一个下标
        testSet.append(trainingSet[randIndex]) #将随机出来的追加到测试集中
        del(trainingSet[randIndex])   #从训练集中删除该下标
    trainMat=[]; trainClasses = [] #训练数据合集 训练数据分类结果
    for docIndex in trainingSet: #对于训练数据
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex])) #每一个语句加入到训练合集中
        trainClasses.append(classList[docIndex]) #每一个结果增加进集合类别中
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses)) #计算贝叶斯参数
    errorCount = 0 #错误计数
    for docIndex in testSet:        #分类测试集
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]: #如果分类不准确
            errorCount += 1
            print "classification error",docList[docIndex]
    print 'the error rate is: ',float(errorCount)/len(testSet)
    #return vocabList,fullText

#使用贝叶斯分类器从个人广告中获取区域倾向
def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True) 
    return sortedFreq[:30]       

def localWords(feed1,feed0):
    import feedparser
    docList=[]; classList = []; fullText =[]
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    top30Words = calcMostFreq(vocabList,fullText)   #remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2*minLen); testSet=[]           #create test set
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is: ',float(errorCount)/len(testSet)
    return vocabList,p0V,p1V

def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**"
    for item in sortedSF:
        print item[0]
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**"
    for item in sortedNY:
        print item[0]
