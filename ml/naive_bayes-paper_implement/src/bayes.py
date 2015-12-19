#coding:utf-8
from numpy import *
import os
import sys
import time
import random
import feedparser
from bayes_train import *
from bayes_rss import *

def textParse(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] 
    
def spamTest() :
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        wordList = textParse(open('../data/email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('../data/email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    print 'vocab_length: ', len(vocabList)
    trainingSet = range(50); testSet=[]           #create test set
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNBd1
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam, selectArgs, joinList = trainNBd1improve(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNBd1(array(wordVector),p0V,p1V,pSpam, selectArgs, joinList) != classList[docIndex]:
            errorCount += 1
            print "classification error",docList[docIndex]
    print 'the error rate is: ',float(errorCount)/len(testSet)
    return float(errorCount)/len(testSet)

def traditionalsmsspamTest() :
    testSetSize = 1000
    docList = []
    classList = []
    datafile = open('../data/smsspamcollection/SMSSpamCollection.txt')
    for line in datafile :
        items = line.split('\t')
        if len(items) != 2 or (items[0] != 'spam' and items[0] != 'ham'):
            continue
        docWords = textParse(items[1])
        docList.append(docWords)
        if items[0] == 'spam' :
            classList.append(1)
        elif items[0] == 'ham' :
            classList.append(0)
    vocabList = createVocabList(docList)
    print 'vocab_length:', len(vocabList)
    docindex = range(len(docList))
    random.shuffle(docindex)
    trainSet = docindex[0 : len(docList) - testSetSize]
    testSet = docindex[len(docList) - testSetSize :]
    trainMat = []
    trainClasses = []
    for i in trainSet :
        trainMat.append(setOfWords2Vec(vocabList, docList[i]))
        trainClasses.append(classList[i])
    p0V, p1V, pSpam = trainNBd1(array(trainMat),array(trainClasses))
    errorCount = 0
    for i in testSet :
        wordVector = setOfWords2Vec(vocabList, docList[i]) 
        if classifyNBd1(array(wordVector),p0V,p1V,pSpam, range(len(wordVector)), {}) != classList[i] :
            errorCount += 1
            print "classification error", docList[i]
        else :
            print "success"
    print 'the error rate is: ', float(errorCount) / len(testSet)

def smsspamTest() :
    testSetSize = 1000
    docList = []
    classList = []
    datafile = open('../data/smsspamcollection/SMSSpamCollection.txt')
    for line in datafile :
        items = line.split('\t')
        if len(items) != 2 or (items[0] != 'spam' and items[0] != 'ham'):
            continue
        docWords = textParse(items[1])
        docList.append(docWords)
        if items[0] == 'spam' :
            classList.append(1)
        elif items[0] == 'ham' :
            classList.append(0)
    vocabList = createVocabList(docList)
    print 'vocab_length:', len(vocabList)
    docindex = range(len(docList))
    random.shuffle(docindex)
    trainSet = docindex[0 : len(docList) - testSetSize]
    testSet = docindex[len(docList) - testSetSize :]
    trainMat = []
    trainClasses = []
    for i in trainSet :
        trainMat.append(setOfWords2Vec(vocabList, docList[i]))
        trainClasses.append(classList[i])
    p0V, p1V, pSpam, selectArgs, joinList = trainNBd1improve(array(trainMat),array(trainClasses))
    ###############test################
    logfile = open('./bayes.log', 'w')
    selectWords = []
    for item in selectArgs :
        selectWords.append(vocabList[item])
    logfile.write(','.join(selectWords))
    logfile.close()
    ##################################
    errorCount = 0
    for i in testSet :
        wordVector = setOfWords2Vec(vocabList, docList[i]) 
        if classifyNBd1(array(wordVector),p0V,p1V,pSpam, selectArgs, joinList) != classList[i] :
            errorCount += 1
            print "classification error", docList[i]
        else :
            print "success"
    print 'the error rate is: ', float(errorCount) / len(testSet)

def testGetTopWords() :
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    getTopWords(ny, sf)

