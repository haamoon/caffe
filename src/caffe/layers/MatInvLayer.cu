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
void MatInvLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	
	const Dtype* input_data = bottom[0]->gpu_data();
	Dtype* output_data = top[0]->mutable_gpu_data();
	
	caffe_copy(N_* offset_, input_data, output_data);
	
	//for (int n = 0; n < N_; ++n) {
	//	//caffe_gpu_strided_add_scalar<Dtype>(offset_, lambda_, dim_ + 1, output_data + offset_ * n);
		//caffe_cpu_inverse<Dtype>(dim_, output_data + offset_ * n);
	//}//
}


template <typename Dtype>
void MatInvLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0]) {
    	return;
  	}
	Dtype* input_diff = bottom[0]->mutable_gpu_diff();
	const Dtype* output_data = top[0]->gpu_data();
    const Dtype* output_diff = top[0]->gpu_diff();
    
    // A' = - B^\top B' B^\top
    for (int n = 0; n < N_; ++n) {
		caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, dim_,
    		dim_, dim_,
    	    (Dtype)-1., output_data + offset_ * n, output_diff + offset_ * n,
    	    (Dtype)0., tmp_buffer_.mutable_gpu_data());
    	    
    	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, dim_,
    		dim_, dim_,
    	    (Dtype)1., tmp_buffer_.gpu_data(), output_data + offset_ * n,
    	    (Dtype)0., input_diff + offset_ * n);    		
	}	
}

}  // namespace caffe
