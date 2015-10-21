#ifndef CAFFE_UTIL_MATH_LAPACK_FUNCTIONS_H_
#define CAFFE_UTIL_MATH_LAPACK_FUNCTIONS_H_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit

#include "glog/logging.h"

#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"

extern "C" {
#include <clapack.h>
}
namespace caffe {

template <typename Dtype>
void caffe_cpu_inverse(const int n, Dtype* X);

}  // namespace caffe

#endif  // CAFFE_UTIL_MATH_LAPACK_FUNCTIONS_H_
