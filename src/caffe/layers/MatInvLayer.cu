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
__global__ void AddLambdaEye(const int nthreads, const Dtype* const bottom_data,
    Dtype* top_data, Dtype lambda, const Dtype* coef, int input_offset, int lda) {
  Dtype add_val = lambda;
  CUDA_KERNEL_LOOP(index, nthreads) {
    int mat_index = index % input_offset;
    int n = index / input_offset;
    if(coef != NULL) {
      add_val = lambda * coef[n];
    }
    
    top_data[index] = (mat_index % (lda + 1) == 0) ?
                      (bottom_data[index] + add_val) : bottom_data[index];
  }
}


template <typename Dtype>
void MatInvLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  Dtype* tmp_data = tmp_buffer_.mutable_gpu_data();
  const Dtype* input_data = bottom[0]->gpu_data();
  int count = bottom[0]->count();

  const Dtype* coef = NULL;
  if(bottom.size() > 1) {
    coef = bottom[1]->gpu_data();
  }
  AddLambdaEye<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, input_data, tmp_data, (Dtype) lambda_, coef, offset_, dim_);
  caffe_gpu_inverse<Dtype>(dim_, tmp_data, top[0]->mutable_gpu_data(), N_);
}

template <typename Dtype>
void MatInvLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  if (!propagate_down[0] && (bottom.size() < 2 or !propagate_down[1])) {
    return;
  }

  Dtype* coef_diff = NULL;
  if(bottom.size() > 1) {
    coef_diff = bottom[1]->mutable_cpu_diff();
  }

	Dtype* input_diff = bottom[0]->mutable_gpu_diff();
	const Dtype* output_data = top[0]->gpu_data();
  const Dtype* output_diff = top[0]->gpu_diff();



  for (int n = 0; n < N_; ++n) {
    // A' = - B^\top B' B^\top
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, dim_,
          dim_, dim_,
    	    (Dtype)-1., output_data + offset_ * n, output_diff + offset_ * n,
    	    (Dtype)0., tmp_buffer_.mutable_gpu_data());
    	    
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, dim_,
          dim_, dim_,
    	    (Dtype)1., tmp_buffer_.gpu_data(), output_data + offset_ * n,
    	    (Dtype)0., input_diff + offset_ * n);

    if(coef_diff != NULL) {
      //\beta' = - \lambda < B', BB>
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, dim_,
                            dim_, dim_,
                            (Dtype) -lambda_, output_data + offset_ * n, output_data + offset_ * n,
                            (Dtype)0., tmp_buffer_.mutable_gpu_data());

      caffe_gpu_dot(offset_, output_data + offset_ * n, tmp_buffer_.gpu_data(), coef_diff + n);
    }
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(MatInvLayer);

}  // namespace caffe
