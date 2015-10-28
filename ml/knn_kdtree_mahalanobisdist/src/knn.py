#!/usr/bin/env python
#coding=utf-8
import os
import sys
import numpy as np
import operator
import time
from kd_tree import *
from m_distance import *

def image2vect(filename) : #读入图像信息
    inputfile = open(filename)
    rvect = np.zeros((1, 1024))
    for i in range(32) :
        line = inputfile.readline()
        for j in range(32) :
            rvect[0, 32*i + j] = int(line[j])
    return rvect

def get_class(test_vect, train_set, train_labels, k) : #原始的knn分类
    trainsize = train_set.shape[0]
    sqDiffMat = (np.tile(test_vect, (trainsize, 1)) - train_set)**2
    sqDist = sqDiffMat.sum(axis = 1)
    distlist = sqDist**0.5
    sortDistIndex = distlist.argsort()
    classCount = {}
    for i in range(k) :
        voteIlabel = train_labels[sortDistIndex[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

def get_class_mdist(test_vect, invcovmatrix, train_set, train_labels, k) : #采用马氏距离的knn分类
    distlist = get_distance1_n_covfix(train_set, invcovmatrix, test_vect)  #马氏距离的计算测试样本与所有训练样本的距离(m_distance.py)
    distlist = np.array(distlist)
    sortDistIndex = distlist.argsort()
    classCount = {}
    for i in range(k) :
        voteIlabel = train_labels[sortDistIndex[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

def get_class_kdtree(kd_tree_root, test_data, train_labels, n) : #采用了kd_tree的knn分类
    nearest_info = search_kd_tree_n(kd_tree_root, test_data, sys.float_info[0], n)#搜索kd_tree, 获取距离最近的n个点信息[dist_list, index_list](kd_tree.py)
    classCount = {}
    for i in nearest_info[1] :
        classCount[train_labels[i]] = classCount.get(train_labels[i], 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

def handwritingclasstest() : #主程序1——测试原始knn算法
    begin_time = time.time()
    train_data_fold = '../digits/trainingDigits/'
    test_data_fold = '../digits/testDigits/'
    train_file_list = os.listdir(train_data_fold)
    test_file_list = os.listdir(test_data_fold)
    train_set = []
    train_labels = []
    for item in train_file_list :
        train_labels.append((int)(item.strip().split('_')[0]))
        train_set.append(list(image2vect(train_data_fold + item)[0]))
    train_set = np.array(train_set)

    test_num = len(test_file_list)
    test_error_num = 0.0
    test_count = 0
    for item in test_file_list :
        test_count += 1
        test_real_label = (int)(item.strip().split('_')[0])
        test_data = image2vect(test_data_fold + item)[0]
        test_label = get_class(test_data, train_set, train_labels, 3)
        if test_real_label != test_label :
            test_error_num += 1
        print '%s:%s(%s)' %(str(test_count), item, str(test_label)) 
    end_time = time.time()
    print 'error_num:' + str(test_error_num)
    print 'test_num:' + str(test_num)
    print 'rate:' + str(test_error_num / test_num)
    print "time_used:" + str(end_time - begin_time)


def handwritingclasstest_mdist() : #主程序2——测试马氏距离
    begin_time = time.time()
    train_data_fold = '../digits/trainingDigits/'
    test_data_fold = '../digits/testDigits/'
    train_file_list = os.listdir(train_data_fold)
    test_file_list = os.listdir(test_data_fold)
    train_set = []
    train_labels = []
    for item in train_file_list :
        train_labels.append((int)(item.strip().split('_')[0]))
        train_set.append(list(image2vect(train_data_fold + item)[0]))
    train_set = np.array(train_set)

    #####先获得训练集协方差矩阵的逆#######
    cov_train_set = np.cov(train_set.transpose())
    invcov_trainset = np.linalg.pinv(cov_train_set)
    #####
    test_num = len(test_file_list)
    test_error_num = 0.0
    test_count = 0
    for item in test_file_list :
        test_count += 1
        test_real_label = (int)(item.strip().split('_')[0])
        test_data = image2vect(test_data_fold + item)[0]
        test_label = get_class_mdist(test_data, invcov_trainset, train_set, train_labels, 3)  #用马氏距离的knn分类
        if test_real_label != test_label :
            test_error_num += 1
        print '%s:%s(%s)' %(str(test_count), item, str(test_label)) 
    end_time = time.time()
    print 'error_num:' + str(test_error_num)
    print 'test_num:' + str(test_num)
    print 'rate:' + str(test_error_num / test_num)
    print "time_used:" + str(end_time - begin_time)

def handwritingclasstest_kdtree() : #主程序3——测试kd_tree
    begin_time = time.time()
    train_data_fold = '../digits/trainingDigits/'
    test_data_fold = '../digits/testDigits/'
    train_file_list = os.listdir(train_data_fold)
    test_file_list = os.listdir(test_data_fold)
    train_set = []
    train_labels = []
    for item in train_file_list :
        train_labels.append((int)(item.strip().split('_')[0]))
        train_set.append(list(image2vect(train_data_fold + item)[0]))
    train_set = np.array(train_set)
    kd_tree_root = kd_tree(None, train_set, range(len(train_labels))) #构造kd_tree(kd_tree.py)
    test_num = len(test_file_list)
    test_error_num = 0.0
    test_count = 0
    for item in test_file_list :
        test_count += 1
        test_real_label = (int)(item.strip().split('_')[0])
        test_data = image2vect(test_data_fold + item)[0] 
        test_label = get_class_kdtree(kd_tree_root, test_data, train_labels, 3) #使用kd_tree的knn分类 
        #####测试k=1时原始的kd_tree搜索算法######
        #train_index = search_kd_tree(kd_tree_root, test_data, sys.float_info[0])[1]
        #test_label = train_labels[train_index]
        if test_real_label != test_label :
            test_error_num += 1
        print '%s:%s(%s)' %(str(test_count), item, str(test_label)) 
    end_time = time.time()
    print 'error_num:' + str(test_error_num)
    print 'test_num:' + str(test_num)
    print 'rate:' + str(test_error_num / test_num)
    print "time_used:" + str(end_time - begin_time)





