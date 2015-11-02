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
void MatMultLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	
	const Dtype* A_data = bottom[0]->gpu_data();
	const Dtype* B_data = bottom[1]->gpu_data();
	Dtype* C_data = top[0]->mutable_gpu_data();
	
	//handle the case both A and B are full matrices: C = AB
	if(!A_is_diag_ && !B_is_diag_) {
		for (int n = 0; n < N_M_; ++n) {
			caffe_gpu_gemm<Dtype>(A_transpose_, CblasNoTrans, D_1_,
    	    	D_3_, D_2_,
    	    	(Dtype)1., A_data + A_offset_ * n, B_data + B_offset_ * n,
    	    	(Dtype)0., C_data + C_offset_ * n);		
		}
	}
	//if A is diagonal we scale each row of B by 
	//coressponding coefficient in diagonal of A
	else if(A_is_diag_ && !B_is_diag_) {
		caffe_copy(N_M_* C_offset_, B_data, C_data);
		for (int n = 0; n < N_M_; ++n) {
			for(int r = 0; r < D_2_; ++r) {	
				caffe_gpu_scal(D_3_, A_data[A_offset_ * n + r], C_data + C_offset_ * n + D_3_ * r); 
			}
		}
	} else if(!A_is_diag_ && B_is_diag_) {
		LOG(FATAL) << "B can not be diagonal while A is not diagonal!";		
	} else {
		caffe_gpu_mul(N_M_ * C_offset_, A_data, B_data, C_data);		
	}
}


template <typename Dtype>
void MatMultLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	
	const Dtype* A_data = bottom[0]->gpu_data();
	const Dtype* B_data = bottom[1]->gpu_data();
    const Dtype* C_diff = top[0]->gpu_diff();
	Dtype* A_diff = bottom[0]->mutable_gpu_diff();
	Dtype* B_diff = bottom[1]->mutable_gpu_diff();	
	
	//Both A and B are full matrices: 
	//A' = C' B^\top
	//B' = A^\top C'
	if(!A_is_diag_ && !B_is_diag_) {
		if(A_transpose_ == CblasNoTrans) {
			for (int n = 0; n < N_M_; ++n) {
				if (propagate_down[0]) {
					caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, D_1_,
						D_2_, D_3_, (Dtype)1., 
						C_diff + C_offset_ * n, B_data + B_offset_ * n,
						(Dtype)0., A_diff + A_offset_ * n);
				}
				if (propagate_down[1]) {
					caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, D_2_,
						D_3_, D_1_, (Dtype)1., 
						A_data + A_offset_ * n, C_diff + C_offset_ * n,
						(Dtype)0., B_diff + B_offset_ * n);
		    	}
		    }
		} else {
			for (int n = 0; n < N_M_; ++n) {
				if (propagate_down[0]) {
					caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, D_2_,
						D_1_, D_3_,
						(Dtype)1., B_data + B_offset_ * n, C_diff + C_offset_ * n,
						(Dtype)0., A_diff + A_offset_ * n);
				}
				if (propagate_down[1]) {
					caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, D_2_,
						D_3_, D_1_,
						(Dtype)1., A_data + A_offset_ * n, C_diff + C_offset_ * n,
						(Dtype)0., B_diff + B_offset_ * n);
				}
		    }
		}		
	}
	else if(A_is_diag_ && !B_is_diag_) {
		caffe_copy(N_M_* C_offset_, C_diff, B_diff);
		for (int n = 0; n < N_M_; ++n) {
			for( int r = 0; r < D_1_; ++r) {
				if (propagate_down[0]) {
					 caffe_gpu_dot(D_3_, C_diff + C_offset_ * n + 
							D_3_ * r, B_data + B_offset_ * n + D_3_ * r, A_diff + A_offset_ * n + r);
				}
				if (propagate_down[1]) {  
					caffe_gpu_scal(D_3_, A_data[A_offset_ * n + r], B_diff + B_offset_ * n + D_3_ * r);
				}
			}			
		}
	} else if(!A_is_diag_ && B_is_diag_) {
		LOG(FATAL) << "B can not be diagonal while A is not diagonal!";	
	} else {
		if (propagate_down[0]) {
			caffe_gpu_mul(N_M_ * C_offset_, C_diff, B_data, A_diff);
		}
		if (propagate_down[1]) {
			caffe_gpu_mul(N_M_ * C_offset_, A_data, C_diff, B_diff);
		}
	}
}

}  // namespace caffe
