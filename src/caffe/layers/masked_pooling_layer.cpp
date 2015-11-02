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
void MaskedPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	
  	MaskedPoolingParameter maskedpooling_param = this->layer_param_.masked_pooling_param();
	max_nseg_ = maskedpooling_param.max_segments();
}

template <typename Dtype>
void MaskedPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  
  vector<int> top_shape;
  top_shape.push_back(bottom[0]->num());
  top_shape.push_back(channels_);
  top_shape.push_back(max_nseg_);
  
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void MaskedPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* image_data = bottom[0]->cpu_data();
  const Dtype* mask_data = bottom[1]->cpu_data();
  const Dtype* seg_inds = bottom[2]->cpu_data();
  const Dtype* seg_nums = bottom[3]->cpu_data();
  
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
 
  for (int i = 0; i < top_count; ++i) {
  	top_data[i] = 0;
  }
  
  // The main loop
  for (int n = 0; n < bottom[0]->num(); ++n) {
  	for (int c = 0; c < channels_; ++c) {
    	//iterate over segments
    	for(int seg = 0; seg < seg_nums[n];) {
    		int start_ind = seg_inds[seg]; 
    		int end_ind = seg_inds[++seg];
    		for(int i = start_ind; i < end_ind; i++) {
    			top_data[seg] += image_data[(int)mask_data[i]]/(end_ind - start_ind);
    		}
    	}
    	top_data += channels_;
    	image_data += top[0]->offset(0,1);
    }  	
    seg_inds += max_nseg_;
  }
}

template <typename Dtype>
void MaskedPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	CHECK(!propagate_down[1]) << "Can not backpropagate to mask";
	CHECK(!propagate_down[2]) << "Can not backpropagate to segment indices";
	CHECK(!propagate_down[3]) << "Can not backpropagate to segment numbers";

	if (!propagate_down[0]) {
		return;
	}
    
	const Dtype* top_diff = top[0]->cpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* seg_nums = bottom[3]->cpu_data();
  	const Dtype* seg_inds = bottom[2]->cpu_data();
	const Dtype* mask_data = bottom[1]->cpu_data();
	
	for (int i = 0; i < bottom[0]->count(); ++i) {
		bottom_diff[i] = 0;
	}
  
	// The main loop
	for (int n = 0; n < bottom[0]->num(); ++n) {
		for (int c = 0; c < channels_; ++c) {
			//iterate over segments
			for(int seg = 0; seg < seg_nums[n];) {
				int start_ind = seg_inds[seg]; 
				int end_ind = seg_inds[++seg];
				for(int i = start_ind; i < end_ind; i++) {
					bottom_diff[(int)mask_data[i]] += 
						top_diff[seg]/(end_ind - start_ind);
				}
			}
			top_diff += channels_;
			bottom_diff += bottom[0]->offset(0,1);
		}  	
		seg_inds += max_nseg_;
	}
}


//#ifdef CPU_ONLY
//STUB_GPU(MaskedPoolingLayer);
//#endif

INSTANTIATE_CLASS(MaskedPoolingLayer);
REGISTER_LAYER_CLASS(MaskedPooling);

}  // namespace caffe
