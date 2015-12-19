#!/usr/bin/env python
#coding:utf-8
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import Image
from numpy import *
from theano import *
from plot_im import *
import time
import cPickle

def load_pic(filepath) :
    #filepath = '../NoisyOffice/SimulatedNoisyOffice/simulated_noisy_images_grayscale/Fontfre_Noisec_TE.png'
    #filepath = '../NoisyOffice/SimulatedNoisyOffice/clean_images_grayscale/Fontfre_Clean_TE.png'
    im = Image.open(filepath)
    imdata = array(im.getdata())
    if not isinstance(imdata[0], int) :
        imdata = imdata[:,0:1]
    imdata = imdata.reshape((im.size[1], im.size[0])) / 255.0
    #plot_im(imdata)
    return imdata

def draw_pic(imdata, filepath) :
    im_shape = (imdata.shape[1], imdata.shape[0])
    imdata = asarray(imdata * 255.0, dtype = 'int32')
    plot_im(imdata)
    imdata = imdata.reshape((im_shape[0] * im_shape[1]))
    im = Image.new('L', im_shape)
    im.putdata(imdata) 
    im.save(filepath)

def load_data(filefold) :
    simu_datafold = filefold + '/SimulatedNoisyOffice'
    simu_noisedatafold = simu_datafold + '/simulated_noisy_images_grayscale/'
    simu_cleandatafold = simu_datafold + '/clean_images_grayscale/'

    real_noisedatafold = filefold + '/RealNoisyOffice/real_noisy_images_grayscale/'
    
    simu_noisefiles = os.listdir(simu_noisedatafold)
    simu_cleanfiles = os.listdir(simu_cleandatafold)
    simu_noisefiles.sort()
    simu_cleanfiles.sort()

    train_noiseset = [[], [], [], []]
    valid_noiseset = [[], [], [], []]
    test_noiseset = [[], [], [], []]
    noiseset = [train_noiseset, valid_noiseset, test_noiseset]

    train_cleanset = []
    valid_cleanset = []
    test_cleanset = []
    cleanset = [train_cleanset, valid_cleanset, test_cleanset]

    noisetype = {'Noisec' : 0, 'Noisef' : 1, 'Noisep' : 2, 'Noisew' : 3}
    datatype = {'TR' : 0, 'VA' : 1, 'TE' : 2}
    for picfile in simu_noisefiles :
        picinfo = picfile.strip().split('.')[0].split('_')
        noiseset[datatype[picinfo[2]]][noisetype[picinfo[1]]].append(load_pic(simu_noisedatafold + picfile))

    for picfile in simu_cleanfiles :
        picinfo = picfile.strip().split('.')[0].split('_')
        cleanset[datatype[picinfo[2]]].append(load_pic(simu_cleandatafold + picfile))
    '''
    shared_noiseset = shared(
            asarray(
                noiseset,
                dtype = config.floatX
                ),
            borrow = True
            )
    shared_cleanset = shared(
            asarray(
                cleanset,
                dtype = config.floatX
                ),
            borrow = True
            )
    '''
    return [noiseset, cleanset]

class denoise_mlp() :
    def __init__(self, noiseset, cleanset, noisetype, window_margin = 5, h1_nodes = 16, h2_nodes = 8, learn_rate = 0.1) :
        self.window_margin = window_margin
        self.h_nodes = [h1_nodes, h2_nodes]
        self.noisetype = noisetype
        self.ilay = tensor.matrix(name = 'ilayer', dtype = config.floatX)
        self.clean_v = tensor.scalar(name = 'clean_value', dtype = config.floatX)
        rng = random.RandomState(123456)
        self.h1_weights = rng.uniform(-2.4/(2*window_margin + 1)**2, 2.4/(2*window_margin + 1)**2,\
                [h1_nodes, 2*window_margin + 1, 2*window_margin + 1])
        self.shared_h1_weights = shared(
                asarray(self.h1_weights, dtype = config.floatX),
                borrow = True
                )
        self.hlay1 = (self.ilay.dimshuffle(('x', 0, 1)) * self.shared_h1_weights).sum(axis = 2).sum(axis = 1)
        self.hlay1 = tensor.nnet.sigmoid(self.hlay1)

        self.h2_weights = rng.uniform(-2.4/h1_nodes, 2.4/h1_nodes, [h2_nodes, h1_nodes])
        self.shared_h2_weights = shared(
                asarray(self.h2_weights, dtype = config.floatX),
                borrow = True
                )
        self.hlay2 = (self.hlay1.dimshuffle(('x', 0)) * self.shared_h2_weights).sum(axis = 1)
        self.hlay2 = tensor.nnet.sigmoid(self.hlay2)

        self.output_weights = rng.uniform(-2.4/h2_nodes, 2.4/h2_nodes, [h2_nodes])
        self.shared_output_weights = shared(
                asarray(self.output_weights, dtype = config.floatX),
                borrow = True
                )
        self.outputlay = (self.hlay2 * self.shared_output_weights).sum(axis = 0)

        self.loss = (self.outputlay - self.clean_v)**2
        
        h1_wg, h2_wg, output_wg = tensor.grad(self.loss, [self.shared_h1_weights, self.shared_h2_weights, self.shared_output_weights])
        updates = (
                (self.shared_h1_weights, self.shared_h1_weights - learn_rate*h1_wg),
                (self.shared_h2_weights, self.shared_h2_weights - learn_rate*h2_wg),
                (self.shared_output_weights, self.shared_output_weights - learn_rate*output_wg)
                )

        self.train_net = function(
                inputs = [self.ilay, self.clean_v],
                outputs = [self.outputlay],
                updates = updates
                )
        
        self.valid_net = function(
                inputs = [self.ilay],
                outputs = [self.outputlay]
                )

        self.test_net = function(
                inputs = [self.ilay],
                outputs = [self.outputlay]
                )

    def denoise_pic(self, im_filepath, output_filepath) :
        im_data = load_pic(im_filepath)
        output_imdata = im_data.copy()
        for loc_x in range(self.window_margin, im_data.shape[0] - self.window_margin) :
            for loc_y in range(self.window_margin, im_data.shape[1] - self.window_margin) :
                output_imdata[loc_x][loc_y] = self.test_net(\
                        im_data[loc_x - self.window_margin : loc_x + self.window_margin + 1,\
                        loc_y - self.window_margin : loc_y + self.window_margin + 1])[0]
        draw_pic(output_imdata, output_filepath)

def build_model(noisetype, epochs = 1, window_margin = 5, h1_nodes = 16, h2_nodes = 8, learn_rate = 0.1) :
    noiseset, cleanset = load_data('../NoisyOffice')
    dn_mlp = denoise_mlp(noiseset, cleanset, noisetype, window_margin, h1_nodes, h2_nodes, learn_rate)
    train_btime = time.time()
    for epoch in range(epochs) :
        print '>>>epoch:%d'%(epoch)
        if len(noiseset[0][noisetype]) != len(cleanset[0]) :
            print 'noiseset and cleanset num not match!'
            break
        for i in range(len(noiseset[0][noisetype])) :
            print '>>pic:%d'%(i)
            if noiseset[0][noisetype][i].shape != cleanset[0][i].shape :
                print 'pic%d:noisepic and cleanpic shape not match!'%(i)
                continue
            pic_shape = noiseset[0][noisetype][i].shape
            for loc_x in range(window_margin, pic_shape[0] - window_margin) :
                for loc_y in range(window_margin, pic_shape[1] - window_margin) :
                    dn_mlp.train_net(noiseset[0][noisetype][i]\
                            [loc_x - window_margin : loc_x + window_margin + 1, loc_y - window_margin : loc_y + window_margin + 1],\
                            cleanset[0][i][loc_x][loc_y])
    print 'train_timeused:%f'%(time.time() - train_btime)
    
    print 'begin to save model...'
    with open('./model_noise%d(%dcycle).pkl'%(noisetype, epochs), 'w') as dumpfile :
        cPickle.dump(dn_mlp, dumpfile)
    print 'saved!'
    return dn_mlp

def test_model() :
    model_filepath = input('input a model filepath:')
    model = cPickle.load(open(model_filepath))
    while True :
        im_filepath = input('input a im filepath:')
        if im_filepath == 'stop' :
            break
        model.denoise_pic(im_filepath, 'test_output.png')


if __name__ == "__main__" :
    #dn_mlp = build_model(3)
    test_model()



