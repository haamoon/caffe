#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/tracking_layers.hpp"
#include "caffe/util/tracker_math.hpp"

namespace caffe {

template <typename Dtype>
void MatTransposeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
  const Dtype* input_data = bottom[0]->gpu_data();
  Dtype* output_data = top[0]->mutable_gpu_data();
  for (int n = 0; n < N_; ++n) {    
    tracker_gpu_transpose(rows_, cols_, input_data, cols_, output_data, rows_);
    input_data += offset_;
    output_data += offset_;
  }

}


template <typename Dtype>
void MatTransposeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype* input_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* output_diff = top[0]->gpu_diff();
  for (int n = 0; n < N_; ++n) {
    tracker_gpu_transpose(cols_, rows_, output_diff, rows_, input_diff, cols_);
    input_diff += offset_;
    output_diff += offset_;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(MatTransposeLayer);
}  // namespace caffe
