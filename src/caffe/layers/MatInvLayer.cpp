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
#include "caffe/util/math_lapack_functions.hpp"

namespace caffe {

template <typename Dtype>
void MatInvLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	// Set lambda value
  	MatInvParameter matinv_param = this->layer_param_.matinv_param();
	lambda_ = matinv_param.lambda();
}

template <typename Dtype>
void MatInvLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    input_shape_ = bottom[0]->shape();  
    CHECK_GE(input_shape_.size(), 2);
	int input_start_axis = input_shape_.size() - 2;
		
	N_ = bottom[0]->count(0, input_start_axis);
	dim_ = input_shape_[input_start_axis];
	CHECK_EQ(dim_, input_shape_[input_start_axis + 1]) << "Input should a be square matrix.";
	
	offset_ = dim_ * dim_;
	
	//Reshaping the temporary buffer
	if(Caffe::mode() == Caffe::GPU) {
		tmp_buffer_shape_ = input_shape_;
	}
	 else {
		tmp_buffer_shape_.clear();
		tmp_buffer_shape_.push_back(dim_);
		tmp_buffer_shape_.push_back(dim_);
  	}
  	
  	tmp_buffer_.Reshape(tmp_buffer_shape_);
  	
  	//Reshaping top
	top[0]->Reshape(input_shape_); 
}

template <typename Dtype>
void MatInvLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	
	const Dtype* input_data = bottom[0]->cpu_data();
	Dtype* output_data = top[0]->mutable_cpu_data();
	
	caffe_copy(N_* offset_, input_data, output_data);
	
	for (int n = 0; n < N_; ++n) {
		caffe_strided_add_scalar<Dtype>(offset_, lambda_, dim_ + 1, output_data + offset_ * n);
		caffe_cpu_inverse<Dtype>(dim_, output_data + offset_ * n);
	}
}


template <typename Dtype>
void MatInvLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0]) {
    	return;
  	}
	Dtype* input_diff = bottom[0]->mutable_cpu_diff();
	const Dtype* output_data = top[0]->cpu_data();
    const Dtype* output_diff = top[0]->cpu_diff();
    
    // A' = - B^\top B' B^\top
    for (int n = 0; n < N_; ++n) {
		caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, dim_,
    		dim_, dim_,
    	    (Dtype)-1., output_data + offset_ * n, output_diff + offset_ * n,
    	    (Dtype)0., tmp_buffer_.mutable_cpu_data());
    	    
    	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, dim_,
    		dim_, dim_,
    	    (Dtype)1., tmp_buffer_.cpu_data(), output_data + offset_ * n,
    	    (Dtype)0., input_diff + offset_ * n);    		
	}	
}

#ifdef CPU_ONLY
STUB_GPU(MatInvLayer);
#endif


INSTANTIATE_CLASS(MatInvLayer);
REGISTER_LAYER_CLASS(MatInv);

}  // namespace caffe
