#!/usr/bin/env python
#coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from numpy import *
from fclayer import *

class SoftmaxOutputLayer(FcLayer) :
    def __init__(self, lay_len, pre_nodesnum) :
        FcLayer.__init__(self, lay_len, pre_nodesnum)
        Fi = pre_nodesnum
        self.weight = random.uniform(-2.4/Fi, 2.4/Fi, [lay_len, pre_nodesnum])

    def softmax(self, pre_maps, node_index = -1) :
        pre_nodes = pre_maps.reshape([pre_maps.shape[0] * pre_maps.shape[1] * pre_maps.shape[2]])
        self.expresult = zeros(self.maps.shape)
        for i in range(self.maps.shape[2]) :
            self.expresult[0][0][i] = exp(dot(pre_nodes, self.weight[i]))
        self.regularbase = sum(self.expresult)
        if node_index != -1 :
            self.maps[0][0][node_index] = self.expresult[0][0][node_index] / self.regularbase 
        else :
            for i in range(self.maps.shape[2]) :
                self.maps[0][0][i] = self.expresult[0][0][i] / self.regularbase
    
    def back_propa(self, pre_mapset, current_error, learn_rate, isweight_update) :
        self.current_error = current_error
        if isweight_update :
            weight_update = dot((self.current_error[0] - self.maps[0]).T, pre_mapset[0])
            self.weight -= learn_rate * weight_update
        target_class = argmax(current_error[0][0])
        pre_error = -(self.weight[target_class] - \
                array([self.weight[i] * self.maps[0][0][i] for i in range(self.maps.shape[2])]).sum(axis = 0))
        return pre_error.reshape(pre_mapset.shape) 


