#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/tracking_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void RowPoolingForward(const int nthreads,
    const Dtype* matrix_data, const Dtype* seg_data, const Dtype* seg_ptr,
    const Dtype* seg_num, int n_mat_elem, int ncol, const Dtype* seg_coef,
    int seg_data_len, int N, int max_nseg, Dtype* top_data) {
  //nthreads = N_ * max_nseg * ncol_
  CUDA_KERNEL_LOOP(index, nthreads) {
  	int tmp = index;
  	int n = tmp % N;
  	int tmp /= N;
  	int col = tmp % ncol;
  	int seg = tmp / ncol;
  	
  	Dtype sum = 0;
 	if(seg < seg_num[n]) {
 		seg_ptr += n * (max_nseg + 1) + seg;
 		int start_ind = seg_ptr[0]; 
    	int end_ind = seg_ptr[1];
    	for(int i = start_ind; i < end_ind; i++) {
    		sum += matrix_data[ n * n_mat_elem 
    				+ (int)seg_data[n * seg_data_len + i] * ncol + col] 
    				* seg_coef[n * seg_data_len + i];
    	}
    }
    top_data[(n * max_nseg + seg) * ncol + col] = sum;
  }
}

template <typename Dtype>
__global__ void RowPoolingBackward(const int nthreads,
    const Dtype* top_diff, const Dtype* seg_data, const Dtype* seg_ptr,
    const Dtype* seg_num, int n_mat_elem, int ncol, int seg_data_len, int max_nseg, 
    Dtype* bottom_diff) {
  
  //nthreads = N_ * ncol_
  CUDA_KERNEL_LOOP(index, nthreads) {
  	int col = index % ncol;
  	int n = index / ncol;
  	
 	//iterate over segments
	for(int seg = 0; seg < seg_num[n]; ++seg) {
		seg_ptr += n * (max_nseg + 1) + seg;
 		int start_ind = seg_ptr[0]; 
    	int end_ind = seg_ptr[1];
    	
    	for(int i = start_ind; i < end_ind; i++) {
    		bottom_diff[n * n_mat_elem + 
    			(int)seg_data[n * seg_data_len + i] * ncol + col += 
				top_diff(n * max_nseg + seg) * ncol + col] * 
				seg_coef[n * seg_data_len + i];
    	}
	}		
  }
}

template <typename Dtype>
void RowPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* matrix_data = bottom[0]->gpu_data();
  const Dtype* seg_data = bottom[1]->gpu_data();
  const Dtype* seg_ptr = bottom[2]->gpu_data();
  const Dtype* seg_num = bottom[3]->gpu_data();
  const Dtype* seg_coef = bottom[4]->gpu_data();
  
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int top_count = top[0]->count();
      
  RowPoolingForward<Dtype><<<CAFFE_GET_BLOCKS(top_count), CAFFE_CUDA_NUM_THREADS>>>(
  	top_count, matrix_data, seg_data, seg_ptr, seg_num, bottom[0]->offset(0,1), 
    ncol_, seg_coef, seg_data_len_, N_, seg_ptr_len - 1, top_data);
    
}

template <typename Dtype>
void RowPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	CHECK(!propagate_down[1]) << "Can not backpropagate to seg_data";
	CHECK(!propagate_down[2]) << "Can not backpropagate to seg_ptr";
	CHECK(!propagate_down[3]) << "Can not backpropagate to seg_num";
	CHECK(!propagate_down[4]) << "Can not backpropagate to seg_coef";

	if (!propagate_down[0]) {
		return;
	}
    
	const Dtype* top_diff = top[0]->gpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	const Dtype* seg_data = bottom[1]->gpu_data();
  	const Dtype* seg_ptr = bottom[2]->gpu_data();
    const Dtype* seg_num = bottom[3]->gpu_data();
	const Dtype* seg_coef = bottom[4]->gpu_data();
	
	caffe_gpu_set(bottom[0]->count(), (Dtype)0.0, bottom_diff);
  
  	int count = N_ * ncol_;
    RowPoolingBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
  	count, top_diff, seg_data, seg_ptr, seg_num, bottom[0]->offset(0,1), 
    ncol_, seg_data_len_, seg_ptr_len_ - 1, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(RowPoolingLayer);

}  // namespace caffe
