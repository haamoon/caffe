from __future__ import division
caffe_root = '/home/erick/Recurrent_caffe/caffe/'

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

max_nseg = 200
max_ntrack = 100
feature_dim = 4096
lmbda = 0.5
alpha = 0.5
t = 8
n = 4

np.random.seed(14344)
X = np.random.uniform(size = (t, n, max_nseg, feature_dim))
V = np.random.uniform(size = (t, n, max_nseg, max_ntrack))
cont = np.zeros((t, n))

net = caffe.Net(prototxt, caffe.TEST)
net.blobs['X'].reshape(*X.shape)
net.blobs['X'].data[...] = X
net.blobs['V'].reshape(*V.shape)
net.blobs['V'].data[...] = V
net.blobs['cont'].reshape(*cont.shape)
net.blobs['cont'].data[...] = cont
net.forward()

Y = np.zeros((t, n, max_nseg, max_ntrack))
Wprev = np.zeros((1, n, max_ntrack, feature_dim))
for i in range(0, n):
  for j in range(0, t):
    Wstart = np.dot(np.dot(V[j][i].T, np.linalg.inv(np.dot(X[j][i], X[j][i].T) + np.multiply(lmbda, np.identity(max_nseg)))), X[j][i])
    Wcont = Wprev if cont[j][i] else Wstart
    Y[j][i] = np.dot(X[j][i], Wcont.T)
    Wprev = np.multiply(1 + alpha * lmbda, Wcont) + np.multiply(alpha, np.dot((Y[j][i] - V[j][i]).T, X[j][i]))

print "difference:", np.linalg.norm(net.blobs['Y'].data - Y)
