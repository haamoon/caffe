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
    
    // See if A or B are diagonal
  	MatMultParameter matmult_param = this->layer_param_.matmult_param();
	int field_size = matmult_param.diagonal_input_size();
	A_is_diag_ = (field_size > 0 && matmult_param.diagonal_input(0));
	B_is_diag_ = (field_size > 1 && matmult_param.diagonal_input(1));
	
	// Does not work if A is full and B is diagonal
	CHECK(A_is_diag_ || !B_is_diag_) << "Full A times diagonal B is not supported.";
	
	// See if we need to transpose A
	A_transpose_ = matmult_param.transpose_a() ? CblasTrans : CblasNoTrans;
}

template <typename Dtype>
void MatMultLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	
	a_shape_ = bottom[0]->shape();
	b_shape_ = bottom[1]->shape();
	
	int a_start_axis = 0;
	int b_start_axis = 0;
	
	if(A_is_diag_) { 
		CHECK_GE(a_shape_.size(), 1);
		a_start_axis = a_shape_.size() - 1;
		D_1_ = a_shape_[a_start_axis];
		D_2_ = D_1_;
		A_offset_ = D_1_;	
	} else {
		CHECK_GE(a_shape_.size(), 2);
		a_start_axis = a_shape_.size() - 2;
		if(A_transpose_ == CblasTrans) {
			D_1_ = a_shape_[a_start_axis + 1];
			D_2_ = a_shape_[a_start_axis];
		} else {
			D_1_ = a_shape_[a_start_axis];
			D_2_ = a_shape_[a_start_axis + 1];
		}
		A_offset_ = D_1_ * D_2_;
	}
	
	if(B_is_diag_) { 
		CHECK_GE(b_shape_.size(), 1);
		b_start_axis = b_shape_.size() - 1;
		CHECK_EQ(D_2_, b_shape_[b_start_axis]) << "Matrices dimension do not match";	
		D_3_ = D_2_;
		B_offset_ = D_2_;
	} else {
		CHECK_GE(a_shape_.size(), 2);
		b_start_axis = b_shape_.size() - 2;
		CHECK_EQ(D_2_, b_shape_[b_start_axis]) << "Matrices dimension do not match";
		D_3_ = b_shape_[b_start_axis + 1];
		B_offset_ = D_2_ * D_3_;
	}
	
	N_M_ = bottom[0]->count(0, a_start_axis);
	CHECK_EQ(N_M_, bottom[1]->count(0, b_start_axis)) << "Num of Matrices should be the same";
	
	
	vector<int> c_shape;
	c_shape.insert(c_shape.end(), a_shape_.begin(), a_shape_.begin() + a_start_axis); 
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
			caffe_cpu_gemm<Dtype>(A_transpose_, CblasNoTrans, D_1_,
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
		if(A_transpose_ == CblasNoTrans) {
			for (int n = 0; n < N_M_; ++n) {
				if (propagate_down[0]) {
					caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, D_1_,
						D_2_, D_3_, (Dtype)1., 
						C_diff + C_offset_ * n, B_data + B_offset_ * n,
						(Dtype)0., A_diff + A_offset_ * n);
				}
				if (propagate_down[1]) {
					caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, D_2_,
						D_3_, D_1_, (Dtype)1., 
						A_data + A_offset_ * n, C_diff + C_offset_ * n,
						(Dtype)0., B_diff + B_offset_ * n);
		    	}
		    }
		} else {
			for (int n = 0; n < N_M_; ++n) {
				if (propagate_down[0]) {
					caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, D_2_,
						D_1_, D_3_,
						(Dtype)1., B_data + B_offset_ * n, C_diff + C_offset_ * n,
						(Dtype)0., A_diff + A_offset_ * n);
				}
				if (propagate_down[1]) {
					caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, D_2_,
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
					A_diff[A_offset_ * n + r] = caffe_cpu_dot(D_3_, C_diff + C_offset_ * n + 
							D_3_ * r, B_data + B_offset_ * n + D_3_ * r);
				}
				if (propagate_down[1]) {  
					caffe_scal(D_3_, A_data[A_offset_ * n + r], B_diff + B_offset_ * n + D_3_ * r);
				}
			}			
		}
	} else if(!A_is_diag_ && B_is_diag_) {
		LOG(FATAL) << "B can not be diagonal while A is not diagonal!";	
	} else {
		if (propagate_down[0]) {
			caffe_mul(N_M_ * C_offset_, C_diff, B_data, A_diff);
		}
		if (propagate_down[1]) {
			caffe_mul(N_M_ * C_offset_, A_data, C_diff, B_diff);
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(MatMultLayer);
#endif


INSTANTIATE_CLASS(MatMultLayer);
REGISTER_LAYER_CLASS(MatMult);

}  // namespace caffe
