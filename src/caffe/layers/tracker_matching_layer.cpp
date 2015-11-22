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
void TrackerMatchingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void TrackerMatchingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(3, bottom[0]->num_axes()) << "Input(0) segment ranks should have 3 axis, "
      << "corresponding to (num, track_id, segment_id)";
  CHECK_EQ(3, bottom[1]->num_axes()) << "Input(1) segment overlaps should have 3 axis, "
      << "corresponding to (num, segment_id, segment_id)";
      
  CHECK_EQ(3, bottom[2]->num_axes()) << "Input(2) segment number should have 1 axis, "
      << "corresponding to (num)";
  
  N_ = bottom[0]->shape(0);
  max_nseg_ = bottom[0]->shape(2);
  max_ntrack_ = bottom[0]->shape(1);
    
  CHECK_EQ(N_, bottom[1]->shape(0));
  CHECK_EQ(N_, bottom[2]->shape(0));
  
  CHECK_EQ(max_nseg_, bottom[1]->shape(0));
  CHECK_EQ(max_nseg_, bottom[1]->shape(1));
  
  
  vector<int> buffer_size;
  buffer_size.push_back(max_ntrack_);
  max_indeces_.Reshape(buffer_size);
  
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void TrackerMatchingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* v_data = bottom[0]->cpu_data();
  const Dtype* overlaps_data = bottom[1]->cpu_data();
  const Dtype* seg_num = bottom[2]->cpu_data();
  
  Dtype* top_data = top[0]->mutable_cpu_data();
  
  int* max_indeces = max_indeces_.mutable_cpu_data();
  
  for(int track = 0; track < max_ntrack_; ++track) {
  	max_indeces[i] = 0;
  }
  
  for(int n = 0; n < N_; ++n) {
    for(int track = 0; track < max_ntrack_; ++track) {
        for(int seg = 1; seg < seg_num[n]; ++seg) {
            if(v_data[*max_indeces] < v_data[seg]) {
                *max_indeces = seg;
  		    }  		
  		    max_indeces += 1;
  		    v_data += max_nseg_;
  		}
  	}
  	max_indeces -= max_ntrack_;
  	for(int track = 0; track < max_ntrack_; ++track) {
        for(int seg = 1; seg < max_nseg; ++seg) {
            top_data[seg] = overlaps_data[*max_indeces * max_nseg_ + seg];    
        }
        *max_indeces = 0;
        max_indeces += 1;
        top_data += max_nseg_;
    }
    max_indeces -= max_ntrack_;
  }
}

#ifdef CPU_ONLY
STUB_GPU(TrackerMatchingLayer);
#endif

INSTANTIATE_CLASS(TrackerMatchingLayer);
REGISTER_LAYER_CLASS(TrackerMatching);

}  // namespace caffe
