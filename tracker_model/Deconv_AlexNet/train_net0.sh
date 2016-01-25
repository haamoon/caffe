#!/usr/bin/env sh

./build/tools/caffe train \
    -solver=./tracker_model/Deconv_AlexNet/solver_frame0.prototxt \
    -weights ./tracker_model/Deconv_AlexNet/fcn-alexnet-pascal.caffemodel \
     2>&1 | tee ./tracker_model/Deconv_AlexNet/t0_caffe_log.txt 
