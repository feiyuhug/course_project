#!/usr/bin/env python
#coding=utf-8
import sys
import numpy as np

#求解向量集中两个向量的马氏距离
def get_distance1_1(vect_set, vect_a, vect_b) :
    matrix = np.array(vect_set)
    vect_a = np.array(vect_a)
    vect_b = np.array(vect_b)
    Tmatrix = matrix.transpose()
    covmatrix = np.cov(Tmatrix)
    invcovmatrix = np.linalg.pinv(covmatrix)
    diff_vect = vect_a - vect_b
    return np.sqrt(np.dot(np.dot(diff_vect, invcovmatrix), diff_vect.transpose()))


#求解测试向量与训练向量集中所有向量的马氏距离
def get_distance1_n(vect_set, test_vect) :
    matrix = np.array(vect_set)
    vect_size = matrix.shape[0]
    test_vect = np.array(test_vect)
    Tmatrix = matrix.transpose()
    covmatrix = np.cov(Tmatrix)
    invcovmatrix = np.linalg.pinv(covmatrix)
    diff_matrix = np.tile(test_vect, (vect_size, 1)) - matrix
    dist_list = []
    for diff_vect in diff_matrix :
        dist_list.append(np.sqrt(np.dot(np.dot(diff_vect, invcovmatrix), diff_vect.transpose())))
    return dist_list


#求解测试向量与训练向量集中所有向量的马氏距离(提前输入伪逆矩阵)
def get_distance1_n_covfix(vect_set, invcovmatrix, test_vect) :
    matrix = np.array(vect_set)
    vect_size = matrix.shape[0]
    test_vect = np.array(test_vect)
    diff_matrix = np.tile(test_vect, (vect_size, 1)) - matrix
    dist_list = []
    for diff_vect in diff_matrix :
        dist_list.append(np.sqrt(np.dot(np.dot(diff_vect, invcovmatrix), diff_vect.transpose())))
    return dist_list

