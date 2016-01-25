#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=./tracker_model/Deconv_AlexNet/solver_xx.prototxt \
    --snapshot=./tracker_model/Deconv_AlexNet/snapshots/xx__iter_2000.solverstate \
    2>&1 | tee ./tracker_model/Deconv_AlexNet/xx_caffe_log.txt
