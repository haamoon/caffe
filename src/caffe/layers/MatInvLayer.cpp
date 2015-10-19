#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/tracking_layers.hpp"

namespace caffe {

template <typename Dtype>
void MatInvLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	//TODO: initialize lambda
}

template <typename Dtype>
void MatInvLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	input_shape_ = bottom[0]->shape();
	CHECK_EQ(input_shape_[1], input_shape_[2]) << "Input should have the same dimentions.";
	top[0]->Reshape(input_shape_); 
}

template <typename Dtype>
void MatInvLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	
	const Dtype* input_data = bottom[0]->cpu_data();
	Dtype* output_data = top[0]->mutable_cpu_data();
	
}


template <typename Dtype>
void MatInvLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	
	const Dtype* input_data = bottom[0]->cpu_data();
	Dtype* input_diff = bottom[0]->mutable_cpu_diff();
	const Dtype* output_data = top[0]->cpu_data();
    const Dtype* output_diff = top[0]->cpu_diff();	
	
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(MatInvLayer, Forward);
STUB_GPU_BACKWARD(MatInvLayer, Backward);
#endif


INSTANTIATE_CLASS(MatInvLayer);
REGISTER_LAYER_CLASS(MatMult);

}  // namespace caffe
