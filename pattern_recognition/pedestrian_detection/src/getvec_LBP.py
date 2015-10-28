#coding:UTF-8
import Image
import sys
import os
reload(sys)

def get_basiclbp(orig) : #对LBP值进行旋转变换，返回旋转之后的最小值
    min_num = orig
    for i in range(7) :
        it = orig % 2
        orig = orig / 2 + it * 128
        if min_num > orig :
            min_num = orig
    return min_num

def get_basiclbplist() : #获取3*3窗口的LBP算子旋转变换后对应的所有模式36个元素的列表
    rlist = []
    for item in range(256) :
        ritem = get_basiclbp(item)
        if ritem not in rlist :
            rlist.append(ritem)
    return rlist

def convert_image(im, x_l, y_l) : #把图像数据转换成灰度矩阵
    cim = []
    linebottom = []
    for y in range(y_l+2) :
        linebottom.append(0)
    cim.append(linebottom)
    for x in range(x_l) :
        line = [0]
        for y in range(y_l) :
            pitem = im.getpixel((x, y))
            if isinstance(pitem, int) :
                line.append(pitem)
            else :
                line.append(pitem[0])
        line.append(0)
        cim.append(line)
    cim.append(linebottom)
    return cim

def get_LBP(cim, x, y) : #计算(x,y)位置处的LBP值
    lbp = 0
    if cim[x][y] < cim[x-1][y-1] :
        lbp = lbp * 2 + 1
    else :
        lbp = lbp * 2
    if cim[x][y] < cim[x-1][y] :
        lbp = lbp * 2 + 1
    else :
        lbp = lbp * 2
    if cim[x][y] < cim[x-1][y+1] :
        lbp = lbp * 2 + 1
    else :
        lbp = lbp * 2
    if cim[x][y] < cim[x][y-1] :
        lbp = lbp * 2 + 1
    else :
        lbp = lbp * 2
    if cim[x][y] < cim[x][y+1] :
        lbp = lbp * 2 + 1
    else :
        lbp = lbp * 2
    if cim[x][y] < cim[x+1][y-1] :
        lbp = lbp * 2 + 1
    else :
        lbp = lbp * 2
    if cim[x][y] < cim[x+1][y] :
        lbp = lbp * 2 + 1
    else :
        lbp = lbp * 2
    if cim[x][y] < cim[x+1][y+1] :
        lbp = lbp * 2 + 1
    else :
        lbp = lbp * 2
    
    return get_basiclbp(lbp)

def get_LBPmap(cim, xl, yl) : #计算LBP图谱
    lbpmap = []
    for x in range(xl) :
        line = []
        for y in range(yl) :
            line.append(get_LBP(cim, x+1, y+1))
        lbpmap.append(line)
    return lbpmap

def get_vects(database_path, v_foldpath) :  #从图像数据库中提取特征按正例负例输出到v_file
    v_file_pos = open(v_foldpath + 'feature_vects.pos', 'w')
    v_file_neg = open(v_foldpath + 'feature_vects.neg', 'w')
    file_names = os.listdir(database_path)
    seg_x = 4  #横向划分区域个数
    seg_y = 8  #纵向划分区域个数
    basic_lbplist = get_basiclbplist()
    basic_lbplist.sort()
    basic_lbpvect = {}
    for item in basic_lbplist :
        basic_lbpvect[item] = 0
    for file_name in file_names: #遍历数据库
        print file_name
        im = Image.open(database_path + '/' + file_name)
        if not im :
            print "fail to open " + file_name
            break
        if file_name.find('pos') != -1 :
            win_x = 12
            win_y = 12
            case_type = True
        elif file_name.find('neg') != -1 :
            win_x = 16
            win_y = 16
            case_type = False
        else :
            print "file_name not in pos or neg!"
            break
        cim = convert_image(im, seg_x * win_x, seg_y * win_y)
        lbpmap = get_LBPmap(cim, seg_x * win_x, seg_y * win_y)
        feature_vector = []
        wlbpvect = basic_lbpvect
        for s_x in range(seg_x) :
            for s_y in range(seg_y) :
                for item in wlbpvect :
                    wlbpvect[item] = 0
                for w_x in range(win_x) :
                    for w_y in range(win_y) :
                        wlbpvect[lbpmap[s_x * win_x + w_x][s_y * win_y + w_y]] += 1
                feature_vector.extend(wlbpvect.values())
        mfea_vect = [('%.30f' %(item / (win_x * win_y + 0.0))) for item in feature_vector]
        if case_type :
            v_file_pos.write(','.join(mfea_vect) + '\n')
        else :
            v_file_neg.write(','.join(mfea_vect) + '\n')
    v_file_pos.close()
    v_file_neg.close()


if __name__ == '__main__' :  #程序入口
    get_vects('../Database_PR_01', './')


