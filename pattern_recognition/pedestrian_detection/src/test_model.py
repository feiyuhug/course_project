#coding:UTF-8
import sys
import os
import numpy as np

test_case_start_index = 701

def load_model(p_filepath, l_filepath) : #加载模型的参数
    pfile = open(p_filepath)
    lfile = open(l_filepath)
    project_vect = []
    for item in pfile :
        project_vect.append(float(item.strip()))
    if len(project_vect) != 1152 :
        print('num of para loaded is error:' + str(len(project_vect)))
        return 0
    limit_info = lfile.read().strip().split('\t')
    limit_point = float(limit_info[0])
    global posflag
    if limit_info[1] == 'True' :
        posflag = True
    elif limit_info[1] == 'False' :
        posflag = False
    else :
        print ('convert str to bool error')
        return 0
    return [project_vect, limit_point]

def load_test_data(pos_file_path, neg_file_path, start_i) : #加载测试集
    global p_input_matrix
    global n_input_matrix
    p_input_matrix = []
    n_input_matrix = []
    pos_file = open(pos_file_path)
    neg_file = open(neg_file_path)
    case_num = 0
    for line in pos_file :
        case_num += 1
	if case_num < start_i :
	    continue
        line = line.strip().split(',')
        line = [float(item) for item in line]
        if len(line) != 1152 :
            print (str(case_num) + ':(pos)input data column_num error!')
            break
        p_input_matrix.append(line)
    case_num = 0
    for line in neg_file :
        case_num += 1
	if case_num < start_i :
	    continue
        line = line.strip().split(',')
        line = [float(item) for item in line]
        if len(line) != 1152 :
            print (str(case_num) + ':(neg)input data column_num error!')
            break
        n_input_matrix.append(line)
    print('test_set:' + str(len(p_input_matrix)))
    pos_file.close()
    neg_file.close()


def run_test() : #测试程序
    pos_r_num = 0
    neg_r_num = 0
    model = load_model('./model', './limit_point')
    load_test_data('./feature_vects.pos', './feature_vects.neg', test_case_start_index)
    proj_vect = np.array([model[0]]).transpose()
    limit = model[1]
    #print(posflag)
    global loc_pos
    loc_pos = []
    global loc_neg
    loc_neg = []
    for vect in p_input_matrix :
        #mvect = [str(item) for item in vect]
        #logfile.write(','.join(mvect) + '\n')
        #mproj = [str(item) for item in proj_vect]
        #logfile.write(','.join(mproj) + '\n')
        #test_result = 0
        #for i in range(len(vect)) :
        #    test_result += vect[i] * proj_vect[i]
        #logfile.write(str(test_result) + '\n')
        loc_pos.append(('%.30f' %np.dot(np.array([vect]), proj_vect)[0][0]))
        if posflag :
            result = np.dot(np.array([vect]), proj_vect)[0][0] + limit 
            if result > 0 :
                pos_r_num += 1
        else :
            if np.dot(np.array([vect]), proj_vect)[0][0] + limit < 0 :
                pos_r_num += 1
    for vect in n_input_matrix :
        loc_neg.append(('%.30f' %np.dot(np.array([vect]), proj_vect)[0][0]))
        if posflag :
            result = np.dot(np.array([vect]), proj_vect)[0][0] + limit
            if result < 0 :
                neg_r_num += 1
        else :
            if np.dot(np.array([vect]), proj_vect)[0][0] + limit > 0 :
                neg_r_num += 1
    print('pos_true:' + str(pos_r_num) + '\n' + 'neg_false:' + str(neg_r_num) + '\n')
    print('rate:' + str((pos_r_num + neg_r_num) / (len(p_input_matrix) + len(n_input_matrix) + 0.0)))

if __name__ == '__main__' :
    conf_file = open('./conf')
    test_case_start_index = int(conf_file.read().strip()) + 1
    run_test()
    case_locfile = open('./testcase.loc', 'w') #把测试数据的投影位置输出到文件testcase.loc中便于下一步可视化
    case_locfile.write(','.join(loc_pos) + '\n')
    case_locfile.write(','.join(loc_neg) + '\n')



