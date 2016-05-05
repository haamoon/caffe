#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/tracking_layers.hpp"
#include "caffe/util/tracker_math.hpp"

namespace caffe {

template <typename Dtype>
void MatTransposeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  vector<int> input_shape = bottom[0]->shape();  
  CHECK_GE(input_shape.size(), 2);
  int input_start_axis = input_shape.size() - 2;
  
  N_ = bottom[0]->count(0, input_start_axis);
  rows_ = input_shape[input_start_axis];
  cols_ = input_shape[input_start_axis + 1];
  
  offset_ = rows_ * cols_;

  
  //Reshaping top
  vector<int> top_shape(input_shape.begin(), input_shape.end() - 2);
  top_shape.push_back(cols_);
  top_shape.push_back(rows_);
  top[0]->Reshape(top_shape); 
}

template <typename Dtype>
void MatTransposeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
  const Dtype* input_data = bottom[0]->cpu_data();
  Dtype* output_data = top[0]->mutable_cpu_data();
  for (int n = 0; n < N_; ++n) {    
    tracker_cpu_transpose(rows_, cols_, input_data, cols_, output_data, rows_);
    input_data += offset_;
    output_data += offset_;
  }

}


template <typename Dtype>
void MatTransposeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype* input_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* output_diff = top[0]->cpu_diff();
  for (int n = 0; n < N_; ++n) {
    tracker_cpu_transpose(cols_, rows_, output_diff, rows_, input_diff, cols_);
    input_diff += offset_;
    output_diff += offset_;
  }
}

#ifdef CPU_ONLY
STUB_GPU(MatTransposeLayer);
#endif


INSTANTIATE_CLASS(MatTransposeLayer);
REGISTER_LAYER_CLASS(MatTranspose);

}  // namespace caffe
