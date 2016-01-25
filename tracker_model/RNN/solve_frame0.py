from __future__ import division
#caffe_root = '../../'  # this file is expected to be in {caffe_root}/tracker_model/Dec_AlexNet
caffe_root = '/home/amirreza/Desktop/I/Research/Paper/CVPR2016/Recurrent_caffe/caffe/'

import sys
import numpy as np
sys.path.insert(0, caffe_root + 'python')
import caffe


# base net -- the learned coarser model
#base_weights = 'fcn-alexnet-pascal.caffemodel'
base_weights = './snapshots/train_t0__iter_6000.caffemodel'

# init
caffe.set_mode_gpu()
caffe.set_device(0)

solver = caffe.SGDSolver('solver_frame0.prototxt')

# do net surgery to set the deconvolution weights for bilinear interpolation
#interp_layers = [k for k in solver.net.params.keys() if 'deconv' in k]
#interp_surgery(solver.net, interp_layers)

# copy base weights for fine-tuning
solver.net.copy_from(base_weights)

# solve straight through -- a better approach is to define a solving loop to
# 1. take SGD steps
# 2. score the model by the test net `solver.test_nets[0]`
# 3. repeat until satisfied
solver.step(80000)
