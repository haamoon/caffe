#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=./tracker_model/Deconv_AlexNet/solver.prototxt \
    --snapshot=./tracker_model/Deconv_AlexNet/snapshots/t2__iter_2000.solverstate \
    2>&1 | tee ./tracker_model/Deconv_AlexNet/t2_caffe_log.txt 
