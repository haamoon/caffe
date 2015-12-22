# Make sure that caffe is on the python path:
caffe_root = './'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

trackerNet = caffe.Net('./tracker_model/train_val_frame0.prototxt', 
                      './tracker_model/tracker.caffemodel',
                          caffe.TRAIN);

