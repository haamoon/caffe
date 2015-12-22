#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/tracking_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void selectLayerForward(const int nthreads,
    const Dtype* const input_data, const Dtype* const overlaps_data,
    const Dtype* const select_data, Dtype* const top_data, int input_offset,
    int overlaps_offset) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / input_offset;
    
    if(select_data[n] == 1) {
    	top_data[index] = input_data[index];
  	} else {
      int offset = index % input_offset;
      top_data[index] = (offset < overlaps_offset) ?
        overlaps_data[overlaps_offset * n + offset] : 0;
  	}
  }
}

template <typename Dtype>
__global__ void selectLayerBackward(const int nthreads,
    const Dtype* const top_diff, const Dtype* select_data,
    Dtype* const bottom_diff, int input_offset) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / input_offset;
    bottom_diff[index] = (select_data[n] == 1) ? top_diff[index] : 0;
  }
}

template <typename Dtype>
void SelectLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* input_data = bottom[0]->gpu_data();
  const Dtype* overlaps_data = bottom[1]->gpu_data();
  const Dtype* select_data = bottom[2]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  
  selectLayerForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, input_data, overlaps_data, select_data, top_data, input_offset_,
        overlaps_offset_);
}

template <typename Dtype>
void SelectLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  //CHECK(!propagate_down[1]) << "Can not propagate to the Select gate!";
    
	const Dtype* select_data = bottom[2]->gpu_data();
	const Dtype* top_diff = top[0]->gpu_diff();
	Dtype* input_diff = bottom[0]->mutable_gpu_diff();
	int count = top[0]->count();
	
	selectLayerBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, select_data, input_diff, input_offset_);
}

INSTANTIATE_LAYER_GPU_FUNCS(SelectLayer);
}  // namespace caffe
