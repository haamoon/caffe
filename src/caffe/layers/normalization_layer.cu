// add by Binbin Xu
// declanxu@gmail.com or declanxu@126.com
// Zhejiang University, State Key Lab of CAD&CG.


#include <algorithm>
#include <cfloat>
#include <vector>

// #include "thrust/device_vector.h"
#include "caffe/util/io.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/tracking_layers.hpp"

namespace caffe {

template <typename Dtype>
void NormalizationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* squared_data = squared_.mutable_gpu_data();
  Dtype normsqr;
  caffe_gpu_powx(N_*D_, bottom_data, Dtype(2), squared_data);
  for (int i=0; i<N_; ++i) {
    caffe_gpu_asum<Dtype>(D_, squared_data+i*D_, &normsqr);
    Dtype scale = (normsqr == 0) ? 0 : Dtype(pow(normsqr, -0.5));
    caffe_gpu_scale<Dtype>(D_, scale , bottom_data+i*D_, top_data+i*D_);
  }
/*
  const Dtype* out = top[0]->cpu_data();
  for (int i=0; i<n; ++i) {
    int ptr = i*d;
    //Dtype tmp = 0.0;
    std::cout << i << ": ";
    for (int j=0; j < d; ++j) {
	//tmp += out[ptr]*out[ptr++];
	std::cout << out[ptr++] << " ";
    }
    std::cout << "\n";
//    LOG(INFO) << i << ": " << tmp;
  }
*/
}
  
template <typename Dtype>
void NormalizationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype a;
  for (int i=0; i<N_; ++i) {
    caffe_gpu_dot(D_, top_data+i*D_, top_diff+i*D_, &a);
    caffe_gpu_scale(D_, a, top_data+i*D_, bottom_diff+i*D_);
    caffe_gpu_sub(D_, top_diff+i*D_, bottom_diff+i*D_, bottom_diff+i*D_);
    caffe_gpu_dot(D_, bottom_data+i*D_, bottom_data+i*D_, &a);
    Dtype scale = (a == 0) ? 0 : Dtype(pow(a, -0.5));
    caffe_gpu_scale(D_, scale, bottom_diff+i*D_, bottom_diff+i*D_);
  }
/*
const Dtype* b = bottom[0]->cpu_data();
for (int i = 0; i < n; i++) {
    std::cout << i << ": ";
    int tmp = i*128;
    for (int j = 0; j < 128; j++) {
	std::cout << b[tmp++] << " ";
    }
    std::cout << "\n:";
}
*/
}

// INSTANTIATE_CLASS(NormalizationLayer);

INSTANTIATE_LAYER_GPU_FUNCS(NormalizationLayer);
}  // namespace caffe
