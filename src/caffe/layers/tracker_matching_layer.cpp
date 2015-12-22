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
  CHECK_GE(bottom[0]->num_axes(), 2) << "Input(0) segment ranks should have at least 2 axis, "
      << "corresponding to (..., track_id, segment_id)";
  CHECK_GE(bottom[1]->num_axes(), 2) << "Input(1) segment overlaps should have at least 2 axis, "
      << "corresponding to (..., segment_id, segment_id)";
      
  CHECK_GE(bottom[2]->num_axes(), 1) << "Input(2) segment number should have at least 1 axis";
  
  int input_start_axis = bottom[0]->num_axes() - 2;
  N_ = bottom[0]->count(0, input_start_axis);

  max_ntrack_ = bottom[0]->shape(input_start_axis);
  max_nseg_ = bottom[0]->shape(input_start_axis + 1);
  CHECK_EQ(N_, bottom[1]->count(0, input_start_axis));
  CHECK_EQ(N_, bottom[2]->count());
  
  CHECK_EQ(max_nseg_, bottom[1]->shape(input_start_axis));
  CHECK_EQ(max_nseg_, bottom[1]->shape(input_start_axis + 1));
  
  if(Caffe::mode() == Caffe::GPU) {
    vector<int> buffer_size;
    buffer_size.push_back(N_);
    buffer_size.push_back(max_ntrack_);
    max_indeces_.Reshape(buffer_size);
  }
  
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void TrackerMatchingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* v_data = bottom[0]->cpu_data();
  const Dtype* overlaps_data = bottom[1]->cpu_data();
  const Dtype* seg_num = bottom[2]->cpu_data();
  
  Dtype* top_data = top[0]->mutable_cpu_data();
  int max_index = 0;
  //LOG(ERROR) << "MATCHING:" << max_ntrack_ << ", " << max_nseg_ << ", " << N_ << ", " << top[0]->count() << ", " << bottom[0]->count();
  for(int n = 0; n < N_; ++n) {
    for(int track = 0; track < max_ntrack_; ++track) {
      max_index = 0;
      for(int seg = 1; seg < seg_num[n]; ++seg) {
        if(v_data[max_index] < v_data[seg]) {
          max_index = seg;
        }
      }
      //LOG(ERROR) << "N = " << n << " track = " << track << " max index = " << max_index;
      caffe_copy(max_nseg_, overlaps_data + max_index * max_nseg_, top_data);
      v_data += max_nseg_;
      top_data += max_nseg_;
    }
    overlaps_data += max_nseg_ * max_nseg_;
  }
}

#ifdef CPU_ONLY
STUB_GPU(TrackerMatchingLayer);
#endif

INSTANTIATE_CLASS(TrackerMatchingLayer);
REGISTER_LAYER_CLASS(TrackerMatching);

}  // namespace caffe
