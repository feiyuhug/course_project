#coding:UTF-8
import sys
import os
import numpy as np

numofcase_train = 700 #用于训练的正例与负例数(默认为800)

def load_data(pos_file_path, neg_file_path) :
    global p_input_matrix
    global n_input_matrix
    p_input_matrix = []
    n_input_matrix = []
    pos_file = open(pos_file_path)
    neg_file = open(neg_file_path)
    case_num = 0
    for line in pos_file :
        case_num += 1
        line = line.strip().split(',')
        line = [float(item) for item in line]
        if len(line) != 1152 :
            print (str(case_num) + ':(pos)input data column_num error!')
            break
        p_input_matrix.append(line)
        if case_num >= numofcase_train :
            break
    case_num = 0
    for line in neg_file :
        case_num += 1
        line = line.strip().split(',')
        line = [float(item) for item in line]
        if len(line) != 1152 :
            print (str(case_num) + ':(neg)input data column_num error!')
            break
        n_input_matrix.append(line)
        if case_num >= numofcase_train :
            break
    pos_file.close()
    neg_file.close()

def get_cov_matrix(input_matrix) :
    m_vect = np.array(input_matrix).sum(axis = 0) / (numofcase_train + 0.0)
    cov_matrix = np.zeros((len(input_matrix[0]), len(input_matrix[0])))
    for line in input_matrix :
        cov_matrix += np.dot(np.array([line - m_vect]).transpose(), np.array([line - m_vect]))
    return cov_matrix

def get_project_dirct() :
    #pos_cov_matrix = np.cov(np.array(p_input_matrix).transpose())
    pos_cov_matrix = get_cov_matrix(p_input_matrix)
    #printmatrix(pos_cov_matrix, './pos_cov_matrix.test')
    #neg_cov_matrix = np.cov(np.array(n_input_matrix).transpose())
    neg_cov_matrix = get_cov_matrix(n_input_matrix)
    #printmatrix(neg_cov_matrix, './neg_cov_matrix.test')
    total_cov_matrix = pos_cov_matrix + neg_cov_matrix
    #printmatrix(total_cov_matrix, './total_cov_matrix.test')
    inv_total_cov_matrix = np.linalg.inv(total_cov_matrix)
    #printmatrix(inv_total_cov_matrix, './inv_total_cov_matrix.test')
    p_m = np.array(p_input_matrix).sum(axis = 0) / (numofcase_train + 0.0)
    #printmatrix(np.array([p_m]), './p_m.test')
    n_m = np.array(n_input_matrix).sum(axis = 0) / (numofcase_train + 0.0)
    #printmatrix(np.array([n_m]), './n_m.test')
    minus_n_p = np.array([p_m - n_m])
    #S_b = np.dot(minus_n_p.transpose(), minus_n_p)
    #w_v, w_matrix = np.linalg.eig(np.dot(inv_total_cov_matrix, S_b))
    #printmatrix(np.array([w_v]), './特征值.test')
    #printmatrix(minus_n_p, './minus_n_p.test')
    return np.dot(inv_total_cov_matrix, minus_n_p.transpose())

def get_limit_point(model_para) :
    p_m = np.array(p_input_matrix).sum(axis = 0) / (numofcase_train + 0.0)
    n_m = np.array(n_input_matrix).sum(axis = 0) / (numofcase_train + 0.0)
    pflag = False
    p_c_proj = np.dot(p_m, model_para)[0]
    n_c_proj = np.dot(n_m, model_para)[0]
    if p_c_proj > n_c_proj : #pflag=True代表正例的投影值大于阈值
        pflag = True
    elif p_c_proj < n_c_proj :
        pflag = False
    else :
        print('p_c_proj == n_c_proj')
    
    minus_n_p = np.array([n_m - p_m])
    str_model_para = [str(item) for item in model_para]
    str_p_m = [str(item) for item in p_m]
    str_n_m = [str(item) for item in n_m]
    #print('model_para:' + ','.join(str_model_para) + '\n')
    #print('p_m:' + ','.join(str_p_m) + '\n')
    #print('n_m:' + ','.join(str_n_m) + '\n')
    print('p_m:' + str(p_c_proj) + '\n')
    print('n_m:' + str(n_c_proj) + '\n')
    return ('%.30f' %(-0.5 * (p_c_proj + n_c_proj))) + '\t' + str(pflag)

def printmatrix(M, filepath) :
    test_file = open(filepath ,'w')
    for line in M :
        sline = [('%.30f' %item) for item in line]
        test_file.write(','.join(sline) + '\n')
    test_file.close()
    
if __name__ == '__main__' :
    config_file = open('./conf')
    numofcase_train = int(config_file.read().strip())
    model_file = open('./model', 'w')
    load_data('./feature_vects.pos', './feature_vects.neg')
    model_para = get_project_dirct()
    model_para = model_para * 1.0e-10
    limit_point = get_limit_point(model_para)
    printmatrix(model_para, './model')
    lfile = open('./limit_point', 'w')
    lfile.write(limit_point)
    #print(limit_point)



