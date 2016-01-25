from __future__ import division
#caffe_root = '../../'  # this file is expected to be in {caffe_root}/tracker_model/Dec_AlexNet
caffe_root = '/home/amirreza/Desktop/I/Research/Paper/CVPR2016/Recurrent_caffe/caffe/'

import sys
import os
import numpy as np
sys.path.insert(0, caffe_root + 'python')
import caffe

os.chdir('..')
# base net -- the learned coarser model
#base_weights = 'fcn-alexnet-pascal.caffemodel'
base_weights = './snapshots/train_iter_20000.caffemodel'
# init
caffe.set_mode_gpu()
caffe.set_device(0)

solver = caffe.SGDSolver('solver.prototxt')

# do net surgery to set the deconvolution weights for bilinear interpolation
#interp_layers = [k for k in solver.net.params.keys() if 'deconv' in k]
#interp_surgery(solver.net, interp_layers)

# copy base weights for fine-tuning
solver.net.copy_from(base_weights)

os.chdir('./test_seg_pooling')