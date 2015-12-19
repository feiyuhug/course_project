#coding:utf-8
from numpy import *
import os
import sys
import time
import random
import feedparser
from bayes_train import *

def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print "the word: %s is not in my Vocabulary!" % word
    return returnVec

def trainNBd1(trainMatrix, trainCategory) :
    numTrainDoc = len(trainMatrix)
    numWords = len(trainMatrix[0])
    numCategory = sum(trainCategory)
    pCategory = numCategory / float(numTrainDoc)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    for i in range(numTrainDoc) :
        if trainCategory[i] == 1 :
            p1Num += trainMatrix[i]
        else :
            p0Num += trainMatrix[i]
    p1Vect = p1Num / float(numCategory)
    p0Vect = p0Num / float(numTrainDoc - numCategory)
    return p0Vect, p1Vect, pCategory

def trainNBd1improve(trainMatrix, trainCategory) :
    discardnum = len(trainMatrix[0]) - 600
    joinpairnum = 10
    begintime = time.time()
    numTrainDoc = len(trainMatrix)
    numWords = len(trainMatrix[0])
    numCategory = sum(trainCategory)
    pCategory = numCategory / float(numTrainDoc)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    pNum = ones(numWords)
    trainMatrix1 = []
    trainMatrix0 = []
    for i in range(numTrainDoc) :
        if trainCategory[i] == 1 :
            p1Num += trainMatrix[i]
            trainMatrix1.append(trainMatrix[i])
        else :
            p0Num += trainMatrix[i]
            trainMatrix0.append(trainMatrix[i])
        pNum += trainMatrix[i]
    p1Vect = p1Num / float(numCategory)
    p1Vects = array([1 - p1Vect, p1Vect])
    p0Vect = p0Num / float(numTrainDoc - numCategory)
    p0Vects = array([1 - p0Vect, p0Vect])
    pVect = pNum / float(numTrainDoc)
    pVects = array([1 - pVect, pVect])
    print 'begin to calculate mutualInfoP_C!'
    mutualInfoP_C = [pCategory * p1Vects[1][i] * (log(p1Vects[1][i]) - log(pVects[1][i])) \ #计算各个特征与类别的互信息
            + pCategory * p1Vects[0][i] * (log(p1Vects[0][i]) - log(pVects[0][i])) \
            + (1 - pCategory) * p0Vects[1][i] * (log(p0Vects[1][i]) - log(pVects[1][i])) \
            + (1 - pCategory) * p0Vects[0][i] * (log(p0Vects[0][i]) - log(pVects[0][i])) for i in range(numWords)]
    print 'mutualInfoP_C finished!'
    discardArgs = list(argsort(array(mutualInfoP_C)))[0 : discardnum]
    discardArgs.append(-1)
    selectArgs = list(argsort(array(mutualInfoP_C)))[discardnum : ] #选择与类别互信息较大的特征

    conditionMutualInfo = zeros([numWords, numWords])
    collectLen1 = len(trainMatrix1) / 100 if len(trainMatrix1) > 10000 else len(trainMatrix1) / 10
    collectLen0 = len(trainMatrix0) / 100 if len(trainMatrix0) > 10000 else len(trainMatrix0) / 10       
    for i in range(len(selectArgs)) :  #计算两两特征间的互信息
        subbegtime = time.time()
        for j in range(len(selectArgs))[i+1:] :
            #subbegtime1 = time.time()
            m1ijNum = ones([2,2])
            m0ijNum = ones([2,2])
            for t in range(collectLen1) :   #在计算两个特征在类别确定下的联合条件概率分布时，便利整个训练集耗时太长，因此便利训练集的一个子集
                m1ijNum[trainMatrix1[t][selectArgs[i]]][trainMatrix1[t][selectArgs[j]]] += 1
            for t in range(collectLen0) :
                m0ijNum[trainMatrix0[t][selectArgs[i]]][trainMatrix0[t][selectArgs[j]]] += 1
            m1ijNum /= float(collectLen1)
            m0ijNum /= float(collectLen0)
            
            for si in range(2) :
                for sj in range(2) :
                    conditionMutualInfo[selectArgs[i]][selectArgs[j]] += pCategory * m1ijNum[si][sj] * \
                            (log(m1ijNum[si][sj]) - log(p1Vects[si][selectArgs[i]] * p0Vects[sj][selectArgs[j]]))
            #print 'conditionMutualInfo:', i, j, conditionMutualInfo[i][j]
            #print 'TT:', time.time() - subbegtime1
        print 'timeuse:', time.time() - subbegtime
    joinList = {}
    for count in range(joinpairnum) : #获取最佳的合并特征对并计算相应的联合概率分布
        joini = -1
        joinj = -1
        while(joini not in selectArgs or joinj not in selectArgs) :
            joinIndex = argmax(conditionMutualInfo)
            joini = joinIndex / numWords
            joinj = joinIndex % numWords
            conditionMutualInfo[joini][joinj] = 0

        m1ijNum = ones([2,2])
        m0ijNum = ones([2,2])
        for t in range(len(trainMatrix1)) :
            m1ijNum[trainMatrix1[t][joini]][trainMatrix1[t][joinj]] += 1
        for t in range(len(trainMatrix0)) :
            m0ijNum[trainMatrix0[t][joini]][trainMatrix0[t][joinj]] += 1
        m1ijNum /= float(len(trainMatrix1))
        m0ijNum /= float(len(trainMatrix0))

        joinList[str(joini) + '_' + str(joinj)] = [m1ijNum, m0ijNum]
        selectArgs.remove(joini)
        selectArgs.remove(joinj)
    endtime = time.time()
    print 'train_time:', endtime - begintime
    return p0Vect, p1Vect, pCategory, selectArgs, joinList

def classifyNBd1(vec2Classify, p0Vec, p1Vec, pClass1, selectArgs, joinList) :
    p1 = sum([log(p1Vec[i]) if vec2Classify[i] == 1 else log(1 - p1Vec[i]) \
            for i in selectArgs]) + log(pClass1)
    p0 = sum([log(p0Vec[i]) if vec2Classify[i] == 1 else log(1 - p0Vec[i]) \
            for i in selectArgs]) + log(1.0 - pClass1)
    for key in joinList.keys() :
        joini, joinj = key.split('_')
        joini = int(joini)
        joinj = int(joinj)
        p1 += log(joinList[key][0][vec2Classify[joini]][vec2Classify[joinj]])
        p0 += log(joinList[key][1][vec2Classify[joini]][vec2Classify[joinj]])
    if p1 > p0 :
        return 1
    else :
        return 0

