from __future__ import division
#caffe_root = '../../'  # this file is expected to be in {caffe_root}/tracker_model/Dec_AlexNet
caffe_root = '/home/amirreza/Desktop/I/Research/Paper/CVPR2016/Recurrent_caffe/caffe/'

import sys
import os
import numpy as np
sys.path.insert(0, caffe_root + 'python')
import caffe

os.chdir('../../..')
# base net -- the learned coarser model
#base_weights = './tracker_model/Deconv_AlexNet/fcn-alexnet-pascal.caffemodel'
base_weights = './tracker_model/Deconv_AlexNet/snapshots/t0__iter_8000.caffemodel'
prototxt = './tracker_model/Deconv_AlexNet/train_val.prototxt'
# init
caffe.set_mode_gpu()
caffe.set_device(0)

net = caffe.Net(prototxt, base_weights, caffe.TEST)


os.chdir('./tracker_model/Deconv_AlexNet/test_result')
