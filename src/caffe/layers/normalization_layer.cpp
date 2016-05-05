// add by Binbin Xu
// declanxu@gmail.com or declanxu@126.com
// Zhejiang University, State Key Lab of CAD&CG.

#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/tracking_layers.hpp"

namespace caffe {

// template <typename Dtype>
// void NormalizationLayer<Dtype>::LayerSetUp(
//   const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
//   Layer<Dtype>::LayerSetUp(bottom, top);
// }

template <typename Dtype>
void NormalizationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  int input_start_axis = bottom[0]->CanonicalAxisIndex(-1);
  N_ = bottom[0]->count(0, input_start_axis);
  D_ = bottom[0]->count(input_start_axis);
  top[0]->ReshapeLike(*bottom[0]);
  squared_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void NormalizationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* squared_data = squared_.mutable_cpu_data();
  caffe_sqr<Dtype>(N_ * D_, bottom_data, squared_data);
  for (int i = 0; i < D_; ++i) {
    Dtype normsqr = caffe_cpu_asum<Dtype>(D_, squared_data+i*D_);
    Dtype scale = (normsqr == 0) ? 0 : Dtype(pow(normsqr, -0.5));
    caffe_cpu_scale<Dtype>(D_, scale, bottom_data+i*D_, top_data+i*D_);
  }
}

template <typename Dtype>
void NormalizationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  for (int i = 0; i < N_; ++i) {
    Dtype a = caffe_cpu_dot(D_, top_data+i*D_, top_diff+i*D_);
    caffe_cpu_scale(D_, a, top_data+i*D_, bottom_diff+i*D_);
    caffe_sub(D_, top_diff+i*D_, bottom_diff+i*D_, bottom_diff+i*D_);
    a = caffe_cpu_dot(D_, bottom_data+i*D_, bottom_data+i*D_);
    Dtype scale = (a == 0) ? 0 : Dtype(pow(a, -0.5));
    caffe_cpu_scale(D_, scale, bottom_diff+i*D_, bottom_diff+i*D_);
  }
}


#ifdef CPU_ONLY
STUB_GPU(NormalizationLayer);
#endif

INSTANTIATE_CLASS(NormalizationLayer);
REGISTER_LAYER_CLASS(Normalization);

}  // namespace caffe
