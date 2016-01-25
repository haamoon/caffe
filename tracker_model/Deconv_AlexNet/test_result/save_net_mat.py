from __future__ import division
#caffe_root = '../../'  # this file is expected to be in {caffe_root}/tracker_model/Dec_AlexNet
caffe_root = '/home/amirreza/Desktop/I/Research/Paper/CVPR2016/Recurrent_caffe/caffe/'

import sys
import os
import numpy as np
sys.path.insert(0, caffe_root + 'python')
import caffe
import scipy.io as sio

prototxt_file = './tracker_model/Deconv_AlexNet/train_val.prototxt'
base_weights = './tracker_model/Deconv_AlexNet/fcn-alexnet-pascal.caffemodel'

os.chdir(caffe_root)

cpu_trackerNet = caffe.Net(prototxt_file,
                          base_weights,
                          caffe.TEST);


caffe.set_mode_cpu()
cpu_trackerNet.forward()

gpu_trackerNet = caffe.Net(prototxt_file,
                           base_weights,
                           caffe.TEST);
caffe.set_mode_gpu()
gpu_trackerNet.forward()

os.chdir('./tracker_model/Deconv_AlexNet/test_result')

# Save data into tracker_test.mat format
alter_names = {'vtilde':'Vtilder', 'v':'Vr', 'seg_feature':'X',
               'seg_num':'seg_nums', 'overlaps':'overlaps', 'cont':'cont'
#For supervised part only
                ,'gt_overlaps':'gt_overlaps', 'gt_num':'gt_nums', 'supervised_loss':'supervised_loss'
               
#               ,'out_best_track':'best_track', 'out_rO':'rO'
                 
##             
#               , 'out_mx_1':'mx_1', 'out_xmx_1':'xmx_1', 'out_m_1':'m_1'
#               , 'out_c_1':'c_1', 'out_c_2':'c_2', 'out_h_1':'h_1', 
#               'out_h_2':'h_2', 'out_w_0':'w_0', 'out_w_1':'w_1', 
#               'out_hm1_0':'hm1_0', 'out_hm1_1':'hm1_1', 'out_m_2':'m_2'
               }
cpu_data = dict()
gpu_data = dict()
tracker_data = dict()

# Save important blobs data
for cpp_name, mat_name in alter_names.iteritems():
  cpu_data[mat_name] = cpu_trackerNet.blobs[cpp_name].data.transpose()
  gpu_data[mat_name] = gpu_trackerNet.blobs[cpp_name].data.transpose()


# Save network parameters
cpu_parameters = {'T': cpu_trackerNet.blobs['cont'].data.shape[0],
              'N': cpu_trackerNet.blobs['cont'].data.shape[1],
              'num_track': cpu_trackerNet.blobs['v'].data.shape[2]}

gpu_parameters = {'T': gpu_trackerNet.blobs['cont'].data.shape[0],
                  'N': gpu_trackerNet.blobs['cont'].data.shape[1],
                  'num_track': gpu_trackerNet.blobs['v'].data.shape[2]}

for name, value in cpu_parameters.iteritems():
  cpu_data[name] = value

for name, value in gpu_parameters.iteritems():
  gpu_data[name] = value

sio.savemat('./cpu_data', cpu_data)
sio.savemat('./gpu_data', gpu_data)


