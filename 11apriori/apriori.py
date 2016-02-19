# -*- coding: utf-8 -*-
'''
Created on Mar 24, 2011
Ch 11 code
@author: Peter
'''
'''
关联分析：
    频繁项集{}:
        支持度 = 包含某项目的项集/总项集数
    关联规则{}->{}:
        可信度 置信度{l1}->{l2} = 支持度(l1,l2)/支持度(l2)
'''
from numpy import *

def loadDataSet(): #载入集合数据
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createC1(dataSet): #构建集合C1 大小为1的所有候选项集的集合
    C1 = []
    for transaction in dataSet: #一次遍历每一行数据
        for item in transaction: #一次遍历每一个数据
            if not [item] in C1: #如果不在C1中
                C1.append([item]) #加入到C1中,每一个元素都是一个list为了frozenset
                
    C1.sort() #排序
    return map(frozenset, C1) #返回的是一个list [frozenset([]),frozenset([])]

def scanD(D, Ck, minSupport):#用于C1->L1 去掉不符合要求的项集
    #候选集合的列表D [set([..])..]
    #数据集Ck：[frozenset([..])..]
    #最小支持度minSupport 
    #c1是所有可以生成的项集 l1是满足minSupport的项集 c1 l1都是第一层的元素
    ssCnt = {}  #创建空字典 {frozenset([..]):1}
    for tid in D: #遍历数据集中的所有组合 set([...])
        for can in Ck: #遍历C1中的所有频繁集合 frozenset([..])
            if can.issubset(tid): #如果是子集
                if not ssCnt.has_key(can): ssCnt[can]=1 #设为1
                else: ssCnt[can] += 1 #增加1
    numItems = float(len(D)) #计算分母 所有项集数
    retList = [] #要返回的L1 [frozenset([..])..]
    supportData = {} #要返回的 支持度值的字典 以备后用 {frozenset([..]):13}
    for key in ssCnt: #ssCnt是个字典 c1中的集合：该集合出现的次数
        support = ssCnt[key]/numItems #计算支持度
        if support >= minSupport: #如果支持度大于min
            retList.insert(0,key) #插入到l1的列表中
        supportData[key] = support  #这个项集的支持度是多少
    return retList, supportData #[frozenset([..])..] {frozenset([..]):13}

def aprioriGen(Lk, k): #频繁项集列表lk-1 项集元素个数k 输出是ck 生成下一层频繁项集
    retList = [] #存放ck [frozenset([..])..]
    lenLk = len(Lk) #上一层频繁项集个数
    for i in range(lenLk): #依次遍历lk-1
        for j in range(i+1, lenLk): #取第二个元素 i<j
        #要合并的是 只有i和j不同的两个元素
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2] #拿出l1和l2两个元素中前k-2个元素 再加上ij正好k个元素
            L1.sort(); L2.sort()
            if L1==L2: #判断前半部分是否相等
                retList.append(Lk[i] | Lk[j]) #合并生成ck中的一个元素
    return retList #返回ck [frozenset([..])..]

def apriori(dataSet, minSupport = 0.5):#输入数据集[[],[],[]]   最小支持度
    #c增加层数的 l满足条件的
    C1 = createC1(dataSet) #首先生成c1 [frozenset([0])..]
    D = map(set, dataSet) #将dataSet转换成集合列表D [set([..]),set([..])...set([..])]
    L1, supportData = scanD(D, C1, minSupport) #[frozenset([..])..] {frozenset([..]):13}
    L = [L1] #[[frozenset([..])..]..]
    k = 2 #要生成第二层
    while (len(L[k-2]) > 0): #还可以扩展
        Ck = aprioriGen(L[k-2], k) #生成ck [frozenset([..])..]
        Lk, supK = scanD(D, Ck, minSupport) #扫描ck生成lk
        supportData.update(supK) #使用一个dict更新另一个字典 其实就是合并 相同则替换
        L.append(Lk) #追加频繁项集
        k += 1 #下一层
    return L, supportData
    #返回频繁项集列表[[1层],[2层],[3层]]  [[frozenset([..])..]..]
    #支持度字典 {frozenset([..]):13}

def generateRules(L, supportData, minConf=0.7):
    #输入：
    #频繁项集列表[1层],[2层],[3层]]  [[frozenset([..])..]..]
    #支持度字典 {frozenset([..]):13}
    #最小可信度值
    bigRuleList = [] #生成一个规则列表，这个列表包含可信度  [(前件,后件,置信度)..]
    for i in range(1, len(L)): #从下标1开始 也就是第二层开始遍历L i:一层频繁项集[frozenset([..])..]
        for freqSet in L[i]: #依次拿出该层的每一个频繁项集 freqSet: frozenset([..])
            H1 = [frozenset([item]) for item in freqSet] #后件：H1:[frozenset([0])..]
            if (i > 1): #如果不是第二层 会进行合并
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else: #如果是第二层 不会进行合并
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

def calcConf(freqSet, H, supportData, brl, minConf=0.7): #计算可信度
    #输入：
    #freqSet：某一层的频繁项集frozenset([..])
    #H:后键列表 [frozenset([0])..]
    #supportData：支持度字典 {frozenset([..]):13}
    #brl：规则列表
    prunedH = [] #[(后件)..]
    for conseq in H: #conseq：frozenset([0]) 依次枚举每一个后件
        conf = supportData[freqSet]/supportData[freqSet-conseq] #计算{包含后件}/{去掉后件}
        if conf >= minConf: #寻找到一个关联规则
            print freqSet-conseq,'-->',conseq,'conf:',conf
            brl.append((freqSet-conseq, conseq, conf)) #追加一个关联规则[(前件,后件,置信度)..]
            prunedH.append(conseq)
    return prunedH #[(后件)..]

#用途：关联规则中生成新的关联规则
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    #输入：
    #freqSet：某一层的频繁项集frozenset([..])
    #H:后键列表 [frozenset([0])..]
    #supportData：支持度字典 {frozenset([..]):13}
    #brl：规则列表
    m = len(H[0]) #初始后件有多少个
    if (len(freqSet) > (m + 1)): #如果后件+1后比频繁项集层数大，说明还有前件
        Hmp1 = aprioriGen(H, m+1) #从后件的列表中生成下一层的后件 使用上述lk-1生成ck的函数，返回[frozenset([..])..]
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf) #计算满足最小可信度条件的后件列表 [(后件)..]
        if (len(Hmp1) > 1):    #如果新生成的新一层后件有不只一个，则可以迭代，合并生成新的规则
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf) #brl返回
            
def pntRules(ruleList, itemMeaning):
    for ruleTup in ruleList: #每一个关联规则
        for item in ruleTup[0]: #前件
            print itemMeaning[item]
        print "           -------->"
        for item in ruleTup[1]: #后件
            print itemMeaning[item]
        print "confidence: %f" % ruleTup[2] #可信度
        print       #print a blank line
        
            
from time import sleep
from votesmart import votesmart
votesmart.apikey = 'a7fa40adec6f4a77178799fae4441030'
#votesmart.apikey = 'get your api key first'
def getActionIds(): #获得数据
    actionIdList = []; billTitleList = []
    fr = open('recent20bills.txt') 
    for line in fr.readlines():
        billNum = int(line.split('\t')[0])
        try:
            billDetail = votesmart.votes.getBill(billNum) #api call
            for action in billDetail.actions:
                if action.level == 'House' and \
                (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
                    actionId = int(action.actionId)
                    print 'bill: %d has actionId: %d' % (billNum, actionId)
                    actionIdList.append(actionId)
                    billTitleList.append(line.strip().split('\t')[1])
        except:
            print "problem getting bill %d" % billNum
        sleep(1)                                      #delay to be polite
    return actionIdList, billTitleList
        
def getTransList(actionIdList, billTitleList): #this will return a list of lists containing ints
    itemMeaning = ['Republican', 'Democratic']#list of what each item stands for
    for billTitle in billTitleList:#fill up itemMeaning list
        itemMeaning.append('%s -- Nay' % billTitle)
        itemMeaning.append('%s -- Yea' % billTitle)
    transDict = {}#list of items in each transaction (politician) 
    voteCount = 2
    for actionId in actionIdList:
        sleep(3)
        print 'getting votes for actionId: %d' % actionId
        try:
            voteList = votesmart.votes.getBillActionVotes(actionId)
            for vote in voteList:
                if not transDict.has_key(vote.candidateName): 
                    transDict[vote.candidateName] = []
                    if vote.officeParties == 'Democratic':
                        transDict[vote.candidateName].append(1)
                    elif vote.officeParties == 'Republican':
                        transDict[vote.candidateName].append(0)
                if vote.action == 'Nay':
                    transDict[vote.candidateName].append(voteCount)
                elif vote.action == 'Yea':
                    transDict[vote.candidateName].append(voteCount + 1)
        except: 
            print "problem getting actionId: %d" % actionId
        voteCount += 2
    return transDict, itemMeaning
