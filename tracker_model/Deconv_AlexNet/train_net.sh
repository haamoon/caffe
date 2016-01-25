#!/usr/bin/env sh

./build/tools/caffe train \
    -solver=./tracker_model/Deconv_AlexNet/solver.prototxt \
    -weights ./tracker_model/Deconv_AlexNet/snapshots/t0__iter_8000.caffemodel \
     2>&1 | tee ./tracker_model/Deconv_AlexNet/t2_caffe_log.txt 
