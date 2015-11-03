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
__global__ void MaskedPoolingForward(const int nthreads,
    const Dtype* image_data, const Dtype* mask_data, const Dtype* seg_inds,
    const Dtype* seg_nums, int n_pixcel, 
    int mask_lenght, int N, int max_nseg, int channels, Dtype* top_data) {
  //nthreads = N_ * channels_ * max_nseg_
  CUDA_KERNEL_LOOP(index, nthreads) {
  	int tmp = index;
  	int seg = tmp % max_nseg;
  	tmp /= max_nseg;
  	int c = tmp % channels;
  	int n = tmp / channels;
  	
 	if(seg < seg_nums[n]) {
 		int start_ind = seg_inds[n * max_nseg + seg]; 
    	int end_ind = seg_inds[n * max_nseg + seg + 1];
    	for(int i = start_ind; i < end_ind; i++) {
    		top_data[(n * max_nseg + seg) * channels + c] += image_data[ (n * channels
    		 + c) * n_pixcel + (int)mask_data[n * mask_lenght + i]]
    		 /(end_ind - start_ind);
    	}
    }
    		
  }
}

template <typename Dtype>
__global__ void MaskedPoolingBackward(const int nthreads,
    const Dtype* top_diff, const Dtype* mask_data, const Dtype* seg_inds,
    const Dtype* seg_nums, int n_pixcel, int mask_lenght, int max_nseg, 
    int channels, Dtype* bottom_diff) {
  
  //nthreads = N_ * channels_
  CUDA_KERNEL_LOOP(index, nthreads) {
  	int c = index % channels;
  	int n = index / channels;
  	
 	//iterate over segments
	for(int seg = 0; seg < seg_nums[n]; ++seg) {
		int start_ind = seg_inds[n * max_nseg + seg]; 
		int end_ind = seg_inds[n * max_nseg + seg + 1];
		for(int i = start_ind; i < end_ind; i++) {
			bottom_diff[(n * channels + c) * n_pixcel + 
				(int)mask_data[n * mask_lenght + i]] += 
			top_diff[(n * max_nseg + seg) * channels + c]/(end_ind - start_ind);
		}
	}		
  }
}

template <typename Dtype>
void MaskedPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* image_data = bottom[0]->gpu_data();
  const Dtype* mask_data = bottom[1]->gpu_data();
  const Dtype* seg_inds = bottom[2]->gpu_data();
  const Dtype* seg_nums = bottom[3]->gpu_data();
  
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int top_count = top[0]->count();
  
  caffe_gpu_set(top_count, (Dtype)0.0, top_data);
    
  MaskedPoolingForward<Dtype><<<CAFFE_GET_BLOCKS(top_count), CAFFE_CUDA_NUM_THREADS>>>(
  	top_count, image_data, mask_data, seg_inds, seg_nums, bottom[0]->offset(0,1), 
    mask_lenght_, bottom[0]->num(), max_nseg_, channels_, top_data);
    
}

template <typename Dtype>
void MaskedPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	CHECK(!propagate_down[1]) << "Can not backpropagate to mask";
	CHECK(!propagate_down[2]) << "Can not backpropagate to segment indices";
	CHECK(!propagate_down[3]) << "Can not backpropagate to segment numbers";

	if (!propagate_down[0]) {
		return;
	}
    
	const Dtype* top_diff = top[0]->gpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* seg_nums = bottom[3]->gpu_data();
  	const Dtype* seg_inds = bottom[2]->gpu_data();
	const Dtype* mask_data = bottom[1]->gpu_data();
	
	caffe_gpu_set(bottom[0]->count(), (Dtype)0.0, bottom_diff);
  
  	int count = bottom[0]->count(0,2);
    MaskedPoolingBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
  	count, top_diff, mask_data, seg_inds, seg_nums, bottom[0]->offset(0,1), 
    mask_lenght_, max_nseg_, channels_, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(MaskedPoolingLayer);

}  // namespace caffe
