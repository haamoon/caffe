from __future__ import division
caffe_root = '/home/amirreza/Desktop/I/Research/Paper/CVPR2016/caffe/'
import time
import scipy.sparse as ssp
import sys
import numpy as np
sys.path.insert(0, caffe_root + 'python')
import caffe
 
#################
num_spatial_cells = 6
#################


prototxt = './seg_pooling.prototxt'
# init
caffe.set_mode_gpu()
caffe.set_device(0)
#caffe.set_mode_cpu()
net = caffe.Net(prototxt, caffe.TRAIN)

print 'Input shape:', net.blobs['in'].data.shape, 'Output shape:', net.blobs['out'].data.shape

np.random.seed(23)

#set in
in_data = np.random.uniform(size=net.blobs['in'].data.shape)
net.blobs['in'].reshape(*in_data.shape)
net.blobs['in'].data[...] = in_data


#set pooling_data, pooling_ind, pooling_ptr
N = reduce(lambda x, y: x*y, net.blobs['in'].data.shape[:-3])
channels = in_data.shape[-3]
max_nrows = net.blobs['pooling_ptr'].data.shape[-1] - 1
max_data_len = net.blobs['pooling_data'].data.shape[-1]
input_ncell = net.blobs['in'].data.shape[-1] * net.blobs['in'].data.shape[-2]

pooling_data = np.zeros((N, max_data_len))
pooling_ind = np.zeros((N, max_data_len))
pooling_ptr = np.zeros((N, max_nrows + 1))
seg_num = np.zeros((N,))
pout_data = np.zeros((N, max_nrows, channels))
in_diff = np.zeros((N,) + in_data.shape[-3:])
out_diff = np.zeros_like(pout_data)
for n in xrange(N):
    nrows = np.random.random_integers(int(max_nrows/num_spatial_cells)) * num_spatial_cells
    
    data_len_hi = np.min([max_data_len, nrows * input_ncell]) / 2
    data_len_low = data_len_hi / 8
    
    data_len = np.random.random_integers(data_len_low, data_len_hi)
    rows = np.random.random_integers(0, nrows - 1, (data_len,))
    cols = np.random.random_integers(0, input_ncell - 1, (data_len,))
    data = np.random.rand(data_len)
    pooling_matrix = np.zeros((nrows, input_ncell))
    pooling_matrix[rows, cols] = data
    pooling_matrix = ssp.csr_matrix(pooling_matrix)
    
    data_len = pooling_matrix.data.size 
    pooling_data[n, 0:data_len] = pooling_matrix.data
    pooling_ind[n, 0:data_len] = pooling_matrix.indices
    pooling_ptr[n, 0:(nrows+1)] = pooling_matrix.indptr
    seg_num[n] = nrows / num_spatial_cells    
    
    pout_data[n, 0:nrows][...] = pooling_matrix * in_data.reshape((N, channels, -1))[n].T
    out_diff[n][...] = np.random.uniform(size=out_diff.shape[-2:])
    in_diff[n][...] = (out_diff[n][0:nrows].T * pooling_matrix).reshape(in_diff[n].shape)

pout_data = pout_data.reshape(*net.blobs['out'].data.shape)
net.blobs['pooling_data'].reshape(*net.blobs['pooling_data'].data.shape)
net.blobs['pooling_ind'].reshape(*net.blobs['pooling_ind'].data.shape)
net.blobs['pooling_ptr'].reshape(*net.blobs['pooling_ptr'].data.shape)
net.blobs['seg_num'].reshape(*net.blobs['seg_num'].data.shape)

net.blobs['pooling_data'].data[...] = pooling_data.reshape(*net.blobs['pooling_data'].data.shape)
net.blobs['pooling_ind'].data[...] = pooling_ind.reshape(*net.blobs['pooling_ind'].data.shape)
net.blobs['pooling_ptr'].data[...] = pooling_ptr.reshape(*net.blobs['pooling_ptr'].data.shape)
net.blobs['seg_num'].data[...] = seg_num.reshape(*net.blobs['seg_num'].data.shape)

#
net.forward()
time.sleep(1)

print 'Forward Error: ', np.abs(pout_data - net.blobs['out'].data).max()
#
#
out_diff = out_diff.reshape(pout_data.shape)
in_diff = in_diff.reshape(in_data.shape)
net.blobs['out'].diff[...] = out_diff
net.backward()

print 'Backward Error: ', np.abs(in_diff - net.blobs['in'].diff).max()
