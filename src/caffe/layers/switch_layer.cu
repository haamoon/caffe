#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/tracking_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SwitchLayerForward(const int nthreads,
    const Dtype* const bottom_data, const Dtype* switch_data,
    Dtype* const top_data, int input_offset, int lda) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int switch_index = index / input_offset;
    
    if(switch_data[switch_index] == 1) {
    	top_data[index] = bottom_data[index];
  	} else {
  		int mat_index = index % input_offset;
  		if(mat_index % (lda + 1) != 0) {
  			top_data[index] = 0;
  		} else {
  			top_data[index] = 1;
  		}
  	}
  }
}

template <typename Dtype>
__global__ void SwitchLayerBackward(const int nthreads,
    const Dtype* const top_diff, const Dtype* switch_data,
    Dtype* const bottom_diff, int input_offset) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int switch_index = index / input_offset;
    bottom_diff[index] = (switch_data[switch_index] == 1) ? top_diff[index] : 0;
  }
}

template <typename Dtype>
void SwitchLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  const Dtype* input_data = bottom[0]->gpu_data();
  const Dtype* switch_data = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  SwitchLayerForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, input_data, switch_data, top_data, input_offset_, D_2_);
}

template <typename Dtype>
void SwitchLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  //CHECK(!propagate_down[1]) << "Can not propagate to the switch gate!";
    
	const Dtype* switch_data = bottom[1]->gpu_data();
	const Dtype* top_diff = top[0]->gpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	int count = top[0]->count();
	
	SwitchLayerBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, switch_data, bottom_diff, input_offset_);
}

INSTANTIATE_LAYER_GPU_FUNCS(SwitchLayer);
}  // namespace caffe
