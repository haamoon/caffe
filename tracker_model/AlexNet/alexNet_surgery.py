# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

trackerNet = caffe.Net('deploy.prototxt', 
                      '../models/bvlc_alexnet/bvlc_alexnet.caffemodel',
                          caffe.TEST);                          
trackerNet.save('tracker.caffemodel');

