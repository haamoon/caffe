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
void SuperpixelPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void SuperpixelPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input(1) image must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  CHECK_EQ(3, bottom[1]->num_axes()) << "Input(2) spixel_data must have 3 axes, "
      << "corresponding to (num, pixel_id, row_num = 0/column_num = 1)";
  CHECK_EQ(2, bottom[2]->num_axes()) << "Input(3) spixel_ptr must have 2 axes, "
      << "corresponding to (num, ptr)";
  CHECK_EQ(1, bottom[3]->num_axes()) << "Input(4) spixel_num must have 1 axes, "
      << "corresponding to (num,)";
  CHECK_EQ(2, bottom[4]->num_axes()) << "Input(5) mask_size must have 2 axes, "
      << "corresponding to (num, height = 0/width = 1)";
     
  
  N_ = bottom[0]->shape(0);
  CHECK_EQ(N_, bottom[1]->shape(0));
  CHECK_EQ(N_, bottom[2]->shape(0));
  CHECK_EQ(N_, bottom[3]->shape(0));
  CHECK_EQ(N_, bottom[4]->shape(0));
  
  channels_ = bottom[0]->channels();
  image_width_ = bottom[0]->width();
  image_height_ = bottom[0]->height();
  
  //bottom[1] is mask with size N_ x spixel_data_len_
  spixel_data_len_ = bottom[1]->shape(1);
  
  
  //bottom[2] has segmens starting indeces with size N_ x spixel_ptr_len_
  spixel_ptr_len_ = bottom[2]->shape(1);
  
  //X_t = top is a (spixel_ptr_len_ - 1) x channels_ matrix
  // We need s+1 pointer to show start,end pairs of spixels in spixel_data
  vector<int> top_shape;
  top_shape.push_back(bottom[0]->num());
  top_shape.push_back(spixel_ptr_len_ - 1);
  top_shape.push_back(channels_);
  
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void SuperpixelPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* image_data = bottom[0]->cpu_data();
  const Dtype* spixel_data = bottom[1]->cpu_data();
  const Dtype* spixel_ptr = bottom[2]->cpu_data();
  const Dtype* spixel_num = bottom[3]->cpu_data();
  const Dtype* mask_size = bottom[4]->cpu_data();
  
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
 
  for (int i = 0; i < top_count; ++i) {
  	top_data[i] = 0;
  }
  
  // The main loop
  for (int n = 0; n < N_; ++n) {
  	Dtype h_ratio = image_height_ / mask_size[0];
  	Dtype w_ratio = image_width_ / mask_size[1]; 
  	for (int c = 0; c < channels_; ++c) {
    	//iterate over superpixels
    	for(int spixel = 0; spixel < spixel_num[n]; ++spixel) {
    		CHECK_LT(spixel_num[n], spixel_ptr_len_) << "Number of superpixels" <<
    						" exceeds the maximum lenght of the ptr";
    		int start_ind = spixel_ptr[spixel]; 
    		int end_ind = spixel_ptr[spixel + 1];
    		for(int i = start_ind; i < end_ind; i++) {
    			int row = (int)(spixel_data[i * 2] * h_ratio);
    			int col = (int)(spixel_data[i * 2 + 1] * w_ratio);
    			top_data[spixel * channels_ + c] += image_data[row * image_width_ + col]/(end_ind - start_ind);
    		}
    	}
    	image_data += bottom[0]->offset(0,1);
    }
    top_data += channels_ * (spixel_ptr_len_ - 1);   	
    spixel_ptr += spixel_ptr_len_;
    spixel_data += spixel_data_len_;
    mask_size += 2;
  }
}

template <typename Dtype>
void SuperpixelPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	CHECK(!propagate_down[1]) << "Can not backpropagate to spixel_data";
	CHECK(!propagate_down[2]) << "Can not backpropagate to spixel_ptr";
	CHECK(!propagate_down[3]) << "Can not backpropagate to spixel_num";
	CHECK(!propagate_down[4]) << "Can not backpropagate to mask_size";
	
	if (!propagate_down[0]) {
		return;
	}
    
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	const Dtype* top_diff = top[0]->cpu_diff();
	const Dtype* spixel_data = bottom[1]->cpu_data();
	const Dtype* spixel_ptr = bottom[2]->cpu_data();
	const Dtype* spixel_num = bottom[3]->cpu_data();
	const Dtype* mask_size = bottom[4]->cpu_data();
	
	for (int i = 0; i < bottom[0]->count(); ++i) {
		bottom_diff[i] = 0;
	}
  
	// The main loop
	for (int n = 0; n < N_; ++n) {
		Dtype h_ratio = image_height_ / mask_size[0];
  		Dtype w_ratio = image_width_ / mask_size[1];
		for (int c = 0; c < channels_; ++c) {
			//iterate over superpixels
			for(int spixel = 0; spixel < spixel_num[n]; ++spixel) {
				int start_ind = spixel_ptr[spixel]; 
    			int end_ind = spixel_ptr[spixel + 1];
    			for(int i = start_ind; i < end_ind; i++) {
    				int row = (int)(spixel_data[i * 2] * h_ratio);
    				int col = (int)(spixel_data[i * 2 + 1] * w_ratio);
					bottom_diff[row * image_width_ + col] += 
						top_diff[spixel * channels_ + c]/(end_ind - start_ind);
				}
			}
			bottom_diff += bottom[0]->offset(0,1);
		}
		top_diff += channels_ * (spixel_ptr_len_ - 1);  	
		spixel_ptr += spixel_ptr_len_;
    	spixel_data += spixel_data_len_;
    	mask_size += 2;
	}
}


#ifdef CPU_ONLY
STUB_GPU(SuperpixelPoolingLayer);
#endif

INSTANTIATE_CLASS(SuperpixelPoolingLayer);
REGISTER_LAYER_CLASS(SuperpixelPooling);

}  // namespace caffe
