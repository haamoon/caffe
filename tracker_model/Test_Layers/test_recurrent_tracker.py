from __future__ import division
caffe_root = '../../'

import sys
import os
import numpy as np
sys.path.insert(0, caffe_root + 'python')
import caffe

prototxt = './recurrent_tracker.prototxt'
# init
caffe.set_mode_gpu()
caffe.set_device(0)
#caffe.set_mode_cpu()
net = caffe.Net(prototxt, caffe.TEST)

max_nseg = net.blobs['X'].shape[2]
max_ntrack = net.blobs['V'].shape[2]
feature_dim = net.blobs['X'].shape[3]
lmbda = 1
alpha = 1
t = net.blobs['X'].shape[0]
n = net.blobs['X'].shape[1]

np.random.seed(14344)
X = np.random.uniform(size = (t, n, max_nseg, feature_dim))
V = np.random.uniform(size = (t, n, max_ntrack, max_nseg))
cont = np.ones((t, n))
cont[0] = 0
cont[10] = 0


net.blobs['X'].reshape(*X.shape)
net.blobs['X'].data[...] = X
net.blobs['V'].reshape(*V.shape)
net.blobs['V'].data[...] = V
net.blobs['cont'].reshape(*cont.shape)
net.blobs['cont'].data[...] = cont
net.forward()

Y = np.zeros((t, n, max_ntrack, max_nseg))
Wprev = np.zeros((max_ntrack, feature_dim))
for i in range(0, t):
  for j in range(0, n):
    Wstart = np.dot(np.dot(V[i, j], np.linalg.inv(np.dot(X[i, j], X[i, j].T) + lmbda * np.identity(max_nseg))), X[i, j])
    Wcont = Wprev if cont[i, j] else Wstart
    Y[i, j] = np.dot(Wcont, X[i, j].T)
    Wprev = (1 - alpha * lmbda) * Wcont - alpha * np.dot((Y[i, j] - V[i, j]), X[i, j])

print 'normalized matrix difference:\n', (net.blobs['Y'].data - Y) / Y
