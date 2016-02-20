# -*- coding: utf-8 -*-
'''
Created on Jun 14, 2011
FP-Growth FP means frequent pattern
the FP-Growth algorithm needs: 
1. FP-tree (class treeNode)
2. header table (use dict)

This finds frequent itemsets similar to apriori but does not 
find association rules.  

@author: Peter
'''
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue #存放节点名称
        self.count = numOccur #存放计数值
        self.nodeLink = None #存放结点link
        self.parent = parentNode #父节点
        self.children = {}  #子结点
    
    def inc(self, numOccur):
        self.count += numOccur
        
    def disp(self, ind=1):
        print '  '*ind, self.name, ' ', self.count #多少个空格 名字 计数
        for child in self.children.values(): #遍历字典中的值
            child.disp(ind+1)

def createTree(dataSet, minSup=1): #创建FP树 dataSet:{frozenset([]):1}
    headerTable = {} #用来保存头指针表 {'r':13}
    #遍历数据集两次
    for trans in dataSet:#trans:frozenset([])
        for item in trans: #'r'
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans] #计算每一个项目的支持度
    for k in headerTable.keys():  #移除不符合最小支持度的值
        if headerTable[k] < minSup: 
            del(headerTable[k]) #移除字典中的值
    freqItemSet = set(headerTable.keys()) #频繁项集('r','s','d')
    #print 'freqItemSet: ',freqItemSet
    if len(freqItemSet) == 0: return None, None  #如果没有符合要求的频繁项集
    for k in headerTable: #k:'r'    headerTable:{'r':13}
        headerTable[k] = [headerTable[k], None] #{'r':[13,树结构]}
    #print 'headerTable: ',headerTable
    retTree = treeNode('Null Set', 1, None) #创建一个空节点
    for tranSet, count in dataSet.items():  #第二次遍历数据集 transSet:frozenset([])  count:1
        localD = {} #每次拿出来一行数据
        for item in tranSet:  #item:'r' tranSet:frozenset(['r'..])
            if item in freqItemSet: #freqItemSet:('r','s','d')
                localD[item] = headerTable[item][0] #localD:{'r':13}
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            #localD={'r':13,'b':20}
            #排序后 localD={'b':20, 'r':13}
            #orderedItems = ['b','r']
            updateTree(orderedItems, retTree, headerTable, count)#加入到树中，并且更新头指针表
    return retTree, headerTable #return tree and header table

def updateTree(items, inTree, headerTable, count):
    #输入：items=['b','r']
    #inTree=treeNode('Null Set', 1, None)
    #headerTable={'r':[13,树结构]}
    if items[0] in inTree.children:#判断最大的点是否在根结点的儿子中
        inTree.children[items[0]].inc(count) #增加计数 儿子增加计数
    else:   #不在根节点的儿子中
        inTree.children[items[0]] = treeNode(items[0], count, inTree) #新建('r',13,par)，加入到该节点儿子中
        if headerTable[items[0]][1] == None: #如果头指针表为空
            headerTable[items[0]][1] = inTree.children[items[0]] #指向新生成的点
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]]) #更新头指针表
    if len(items) > 1:#如果这一行数据还有其他的节点
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)
        
def updateHeader(nodeToTest, targetNode):   #输入树结构，树结构
    while (nodeToTest.nodeLink != None):    #找到最后一个list
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode #加入目标点
#这个程序写的好烂       
def ascendTree(leafNode, prefixPath): #从节点迭代上溯整棵树
    if leafNode.parent != None: #如果还有父节点
        prefixPath.append(leafNode.name) #返回结果追加当前节点，这是个倒序，叶子->根 none没有加入到其中
        ascendTree(leafNode.parent, prefixPath) #上溯
    
def findPrefixPath(basePat, treeNode): #从头指针表寻找条件模式基
    condPats = {} #存放条件模式基
    while treeNode != None: #
        prefixPath = [] #存放前缀路径
        ascendTree(treeNode, prefixPath) #寻找该节点的前缀路径
        if len(prefixPath) > 1:  #如果前缀路径长度大于1
            condPats[frozenset(prefixPath[1:])] = treeNode.count #{(不包括自己):当前计数}
        treeNode = treeNode.nodeLink #继续走下一条
    return condPats #{frozenset(不包括自己):当前计数}

def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    #inTree:之前创建好的FP树
    #headTable:之间创建好的头指针表
    #minSup:给定的最小支持度
    #preFix:set([])用来存放频繁项集
    #freqItemsList:[] 频繁项集列表
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1])]#正序排列 ['r','b']小的在前
    for basePat in bigL:
        newFreqSet = preFix.copy() #拷贝一份条件模式基
        newFreqSet.add(basePat) #追加一个当前的项
        #print 'finalFrequent Item: ',newFreqSet    #append to set
        freqItemList.append(newFreqSet) #把一个频繁项添加到频繁项集列表中
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1]) #寻找这一个列表的条件模式基 {(不包括自己):当前计数}
        #print 'condPattBases :',basePat, condPattBases
        #2. construct cond FP-tree from cond. pattern base
        myCondTree, myHead = createTree(condPattBases, minSup) #使用条件模式基创建条件FP树
        #print 'head from conditional tree: ', myHead
        if myHead != None: #3. mine cond. FP-tree #如果创建出来了新的条件FP树
            #print 'conditional tree for: ',newFreqSet
            #myCondTree.disp(1)            
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList) #递归挖掘频繁项集

def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat#[[]]

def createInitSet(dataSet): #[[]]
    retDict = {}
    for trans in dataSet: #trans:[]
        retDict[frozenset(trans)] = 1
    return retDict #{frozenset([]):1}

#minSup = 3
#simpDat = loadSimpDat()
#initSet = createInitSet(simpDat)
#myFPtree, myHeaderTab = createTree(initSet, minSup)
#myFPtree.disp()
#myFreqList = []
#mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)

'''
import twitter
from time import sleep
import re

def textParse(bigString):
    urlsRemoved = re.sub('(http:[/][/]|www.)([a-z]|[A-Z]|[0-9]|[/.]|[~])*', '', bigString)    
    listOfTokens = re.split(r'\W*', urlsRemoved)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def getLotsOfTweets(searchStr):
    CONSUMER_KEY = ''
    CONSUMER_SECRET = ''
    ACCESS_TOKEN_KEY = ''
    ACCESS_TOKEN_SECRET = ''
    api = twitter.Api(consumer_key=CONSUMER_KEY, consumer_secret=CONSUMER_SECRET,
                      access_token_key=ACCESS_TOKEN_KEY, 
                      access_token_secret=ACCESS_TOKEN_SECRET)
    #you can get 1500 results 15 pages * 100 per page
    resultsPages = []
    for i in range(1,15):
        print "fetching page %d" % i
        searchResults = api.GetSearch(searchStr, per_page=100, page=i)
        resultsPages.append(searchResults)
        sleep(6)
    return resultsPages

def mineTweets(tweetArr, minSup=5):
    parsedList = []
    for i in range(14):
        for j in range(100):
            parsedList.append(textParse(tweetArr[i][j].text))
    initSet = createInitSet(parsedList)
    myFPtree, myHeaderTab = createTree(initSet, minSup)
    myFreqList = []
    mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
    return myFreqList
'''