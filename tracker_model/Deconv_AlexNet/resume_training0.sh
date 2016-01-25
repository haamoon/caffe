#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=./tracker_model/Deconv_AlexNet/solver_frame0.prototxt \
    --snapshot=./tracker_model/Deconv_AlexNet/snapshots/t0__iter_8000.solverstate \
    2>&1 | tee ./tracker_model/Deconv_AlexNet/t0_caffe_log.txt 
