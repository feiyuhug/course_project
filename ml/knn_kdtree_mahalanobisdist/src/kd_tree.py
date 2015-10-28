#!/usr/bin/env python
#coding=utf-8
import sys
import os
import numpy as np
import copy

def kd_tree(parent, train_set, train_index) : #构造kd_tree，迭代构造，参数(父节点,分支的数据集,数据集在原始数据集中的索引)
    k = len(train_index)
    if k == 0 :
        return None
    elif k == 1 :
        return [parent, None, None, [train_set[0], train_index[0]], None]
    split_axis = np.argmax(np.var(train_set, axis = 0)) #计算数据集各个分量的方差，取方差最大的分量作为切分轴
    split_index = np.argsort(train_set[:,split_axis])
    l_train_set = []
    l_train_index = []
    r_train_set = []
    r_train_index = []
    node_data = [train_set[split_index[k/2]], train_index[split_index[k/2]]] #节点对应样本的特征向量与索引信息
    for i in split_index[0:k/2] :
        l_train_set.append(train_set[i])
        l_train_index.append(train_index[i])
    for i in split_index[k/2 + 1:k] :
        r_train_set.append(train_set[i])
        r_train_index.append(train_index[i])
    split_face = train_set[split_index[k/2]][split_axis] 
    split_info = [split_axis, split_face] #划分信息：划分轴与划分面（分界值）
    node = [parent, None, None, node_data, split_info]
    node[1] = kd_tree(node, np.array(l_train_set), l_train_index) #递归求子节点
    node[2] = kd_tree(node, np.array(r_train_set), r_train_index)
    return node

def search_kd_tree(inode, test_data, cmpdist) : #原始的kd_tree搜索——只能搜索一个最临近的样本点
    search_trace = []
    iterinode = inode
    while(iterinode != None) : #深度优先搜索，沿路压栈
        search_trace.append(iterinode)
        split_info = iterinode[4]
        if split_info == None :
            break
        if test_data[split_info[0]] < split_info[1] :
            iterinode = iterinode[1]
        else :
            iterinode = iterinode[2]
    if not search_trace:
        return None
    nearest_dist = sys.float_info[0] #最近邻距离
    nearest_case_index = -1          #最近邻样本索引
    test_data = np.array(test_data)
    while(search_trace) :     #回溯
        iterinode = search_trace.pop()
        meet_vect = np.array(iterinode[3][0])
        current_dist = np.sum((meet_vect - test_data)**2)**0.5
        if current_dist < nearest_dist : #判断当前节点是否是更近邻
            nearest_dist = current_dist
            nearest_case_index = iterinode[3][1]
        split_info = iterinode[4]
        if split_info == None :
            continue
        if abs(test_data[split_info[0]] - split_info[1]) < nearest_dist : #判断是否需要搜索另一区域
            if test_data[split_info[0]] < split_info[1] :
                current_dist_info = search_kd_tree(iterinode[2], test_data, nearest_dist) #递归搜索
                if current_dist_info != None :
                    nearest_dist = current_dist_info[0]
                    nearest_case_index = current_dist_info[1]
            else :
                current_dist_info = search_kd_tree(iterinode[1], test_data, nearest_dist)
                if current_dist_info != None :
                    nearest_dist = current_dist_info[i0]
                    nearest_case_index = current_dist_info[1]
    if nearest_dist < cmpdist :
        return [nearest_dist, nearest_case_index]
    else :
        return None


def join_nearest_array(dist_list_a, dist_index_list_a, dist_list_b, dist_index_list_b) : #合并两个最近邻的节点信息列表
    dist_list = list(dist_list_a)
    dist_list.extend(list(dist_list_b))
    dist_index_list = list(dist_index_list_a)
    dist_index_list.extend(list(dist_index_list_b))
    sort_index = np.argsort(dist_list)
    top_dist_list = []
    top_dist_index_list = []
    for i in range(len(dist_list_a)) :
        top_dist_list.append(dist_list[sort_index[i]])
        top_dist_index_list.append(dist_index_list[sort_index[i]])
    return [top_dist_list, top_dist_index_list]

def search_kd_tree_n(inode, test_data, cmpdist, n) : #改进后的kd_tree搜索——能一次搜索出n个最近邻
    search_trace = []
    iterinode = inode
    while(iterinode != None) : #深度优先搜索、压栈
        search_trace.append(iterinode)
        split_info = iterinode[4]
        if split_info == None :
            break
        if test_data[split_info[0]] < split_info[1] :
            iterinode = iterinode[1]
        else :
            iterinode = iterinode[2]
    if not search_trace:
        return None
    nearest_dist_list = np.array([sys.float_info[0]]*n) #最近邻距离列表
    nearest_case_index_list = np.array([-1]*n)          #最近邻样本索引列表
    test_data = np.array(test_data)
    while(search_trace) :  #回溯
        iterinode = search_trace.pop()
        meet_vect = np.array(iterinode[3][0])
        current_dist = np.sum((meet_vect - test_data)**2)**0.5
        if current_dist < max(nearest_dist_list) :  #判断当前节点是否也是近邻之一
            mindex = np.argmax(nearest_dist_list)
            nearest_dist_list[mindex] = current_dist
            nearest_case_index_list[mindex] = iterinode[3][1]
        split_info = iterinode[4]
        if split_info == None :
            continue
        mindex = np.argmax(nearest_dist_list)
        if abs(test_data[split_info[0]] - split_info[1]) < nearest_dist_list[mindex] : #这里是用近邻列表中的最大值来判断是否搜索另一区域
            if test_data[split_info[0]] < split_info[1] :
                current_dist_info = search_kd_tree_n(iterinode[2], test_data, nearest_dist_list[mindex], n)
                if current_dist_info != None :
                    nearest_dist_info = join_nearest_array(nearest_dist_list, nearest_case_index_list, current_dist_info[0], current_dist_info[1])  #列表合并
                    nearest_dist_list = nearest_dist_info[0]
                    nearest_case_index_list = nearest_dist_info[1]
            else :
                current_dist_info = search_kd_tree_n(iterinode[1], test_data, nearest_dist_list[mindex], n)
                if current_dist_info != None :
                    nearest_dist_info = join_nearest_array(nearest_dist_list, nearest_case_index_list, current_dist_info[0], current_dist_info[1])
                    nearest_dist_list = nearest_dist_info[0]
                    nearest_case_index_list = nearest_dist_info[1]
    if min(nearest_dist_list) < cmpdist : #只有列表中存在小于cmpdist的值时，列表才有意义，需要返回
        return [nearest_dist_list, nearest_case_index_list]
    else :
        return None




