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
void MatMultLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

}

template <typename Dtype>
void MatMultLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	a_shape_ = bottom[0]->shape();
	b_shape_ = bottom[1]->shape();
	CHECK_GE(a_shape_.size(), 2);
	CHECK_GE(b_shape_.size(), 2);

	N_M_ = a_shape_[0];
	CHECK_EQ(N_M_, b_shape_[0]) << "Num of Matrices should be the same";
	
	
	D_1_ = a_shape_[1];
	
	A_is_diag_ = (a_shape_.size() == 2);
	B_is_diag_ = (b_shape_.size() == 2);
	
	if(A_is_diag_) {
		D_2_ = D_1_;
		A_offset_ = D_1_;
	} else {
		D_2_ = a_shape_[2];
		A_offset_ = D_1_ * D_2_;
	}
	CHECK_EQ(D_2_, b_shape_[1]) << "Matrices dimension do not match";
	
	if(B_is_diag_) {
		D_3_ = D_2_;
		B_offset_ = D_2_;
	} else {
		D_3_ = b_shape_[2];
		B_offset_ = D_2_ * D_3_;
	}
	
	vector<int> c_shape;
	c_shape.push_back(N_M_);
	c_shape.push_back(D_1_);
	
	
	if(A_is_diag_ && B_is_diag_) {
		C_offset_ = D_1_; 
	} else {
		C_offset_ = D_1_ * D_3_;
		c_shape.push_back(D_3_);
	}
	top[0]->Reshape(c_shape); 
}

template <typename Dtype>
void MatMultLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	
	const Dtype* A_data = bottom[0]->cpu_data();
	const Dtype* B_data = bottom[1]->cpu_data();
	Dtype* C_data = top[0]->mutable_cpu_data();
	
	//handle the case both A and B are full matrices: C = AB
	if(!A_is_diag_ && !B_is_diag_) {
		for (int n = 0; n < N_M_; ++n) {
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, D_1_,
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
				caffe_scal(D_3_, A_data[A_offset_ * n + r], C_data + C_offset_ * n + D_3_ * r); 
			}
		}
	} else if(!A_is_diag_ && B_is_diag_) {
		LOG(FATAL) << "B can not be diagonal while A is not diagonal!";		
	} else {
		caffe_mul(N_M_ * C_offset_, A_data, B_data, C_data);		
	}
}


template <typename Dtype>
void MatMultLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	
	const Dtype* A_data = bottom[0]->cpu_data();
	const Dtype* B_data = bottom[1]->cpu_data();
    const Dtype* C_diff = top[0]->cpu_diff();
	Dtype* A_diff = bottom[0]->mutable_cpu_diff();
	Dtype* B_diff = bottom[1]->mutable_cpu_diff();	
	
	//Both A and B are full matrices: 
	//A' = C' B^\top
	//B' = A^\top C'
	if(!A_is_diag_ && !B_is_diag_) {
		for (int n = 0; n < N_M_; ++n) {
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, D_1_,
        		D_2_, D_3_,
        		(Dtype)1., C_diff + C_offset_ * n, B_data + B_offset_ * n,
        		(Dtype)0., A_diff + A_offset_ * n);

			caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, D_2_,
        		D_3_, D_1_,
        		(Dtype)1., A_data + A_offset_ * n, C_diff + C_offset_ * n,
        		(Dtype)0., B_diff + B_offset_ * n);
        }		
	}
	else if(A_is_diag_ && !B_is_diag_) {
		caffe_copy(N_M_* C_offset_, C_diff, B_diff);
		for (int n = 0; n < N_M_; ++n) {
			for( int r = 0; r < D_1_; ++r) {
				A_diff[A_offset_ * n + r] = caffe_cpu_dot(D_3_, C_diff + C_offset_ * n + 
						D_3_ * r, B_data + B_offset_ * n + D_3_ * r);  
				caffe_scal(D_1_, A_data[A_offset_ * n + r], B_diff + B_offset_ * n + D_3_ * r);
			}			
		}
	} else if(!A_is_diag_ && B_is_diag_) {
		LOG(FATAL) << "B can not be diagonal while A is not diagonal!";	
	} else {
		caffe_mul(N_M_ * C_offset_, C_diff, B_data, A_diff);
		caffe_mul(N_M_ * C_offset_, A_data, C_diff, B_diff);
	}
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(MatMultLayer, Forward);
STUB_GPU_BACKWARD(MatMultLayer, Backward);
#endif


INSTANTIATE_CLASS(MatMultLayer);
REGISTER_LAYER_CLASS(MatMult);

}  // namespace caffe
