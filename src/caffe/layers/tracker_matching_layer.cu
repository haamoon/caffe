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
__global__ void kernel_row_max(const int num, const int max_ntrack,
    const int max_nseg, const Dtype* v_data, const Dtype* seg_num, int* max_indeces) {
  CUDA_KERNEL_LOOP(index, num * max_ntrack) {
    int n = index / max_ntrack;
    int track = index % max_ntrack;
    max_indeces += index;
    *max_indeces = 0; 
    v_data += (n * max_ntrack + track) * max_nseg;
    for (int seg = 1; seg < seg_num[n]; ++seg) {
      if(v_data[seg] < v_data[*max_indeces]) {
        *max_indeces = seg;
      }
    }
  }
}


template <typename Dtype>
__global__ void kernel_onehot_product(const int num, const int max_ntrack,
    const int max_nseg, const int* max_indeces, const Dtype* overlap_data, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, num * max_ntrack * max_nseg) {
 
    int seg = index % max_nseg;
    int tmp = index / max_nseg;
    int track = tmp % max_ntrack;   
    int n = index / max_ntrack;
    max_indeces += n * max_ntrack + track; 
    top_data[index] = overlap_data[(n * max_ntrack + *max_indeces) * max_nseg + seg];
  }
}

template <typename Dtype>
void TrackerMatchingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) { 
  
  const Dtype* v_data = bottom[0]->gpu_data();
  const Dtype* overlaps_data = bottom[1]->gpu_data();
  const Dtype* seg_num = bottom[2]->gpu_data();  
	int* max_indeces = max_indeces_.mutable_gpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	
	kernel_row_max<Dtype><<<CAFFE_GET_BLOCKS(N_ * max_ntrack_),
      CAFFE_CUDA_NUM_THREADS>>>(N_, max_ntrack_, max_nseg_, v_data, seg_num,
      max_indeces);
  
  	kernel_onehot_product<Dtype><<<CAFFE_GET_BLOCKS(N_ * max_ntrack_ * max_nseg_),
      CAFFE_CUDA_NUM_THREADS>>>(N_, max_ntrack_, max_nseg_, max_indeces, 
      overlaps_data, top_data);
}

INSTANTIATE_LAYER_GPU_FUNCS(TrackerMatchingLayer);

}  // namespace caffe
