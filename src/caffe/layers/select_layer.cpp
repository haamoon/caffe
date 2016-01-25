#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/tracking_layers.hpp"

namespace caffe {
  
  template <typename Dtype>
  void SelectLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    
    vector<int> input_shape = bottom[0]->shape();
    vector<int> overlaps_shape = bottom[1]->shape();
    vector<int> Select_shape = bottom[2]->shape();
    
    CHECK_GE(input_shape.size(), 2);
    int input_start_axis = input_shape.size() - 2;
    
    num_track_ = input_shape[input_start_axis];
    num_seg_ = input_shape[input_start_axis + 1];
    input_offset_ = num_track_ * num_seg_;
    overlaps_offset_ = num_seg_ * num_seg_;
    
    N_ = bottom[0]->count(0, input_start_axis);
    CHECK_EQ(N_, bottom[1]->count(0, input_start_axis));
    CHECK_EQ(num_seg_, bottom[1]->shape(input_start_axis));
    CHECK_EQ(num_seg_, bottom[1]->shape(input_start_axis + 1));
    CHECK_EQ(N_, bottom[2]->count());
    
    
    //Reshaping top
    top[0]->ReshapeLike(*bottom[0]);
  }
  
  template <typename Dtype>
  void SelectLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    
    const Dtype* input_data = bottom[0]->cpu_data();
    const Dtype* overlaps_data = bottom[1]->cpu_data();
    const Dtype* select_data = bottom[2]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    
    
    for (int n = 0; n < N_; ++n) {
      switch((int)select_data[n]) {
        //set output to the predifined value
        //clip and forward overlaps_data (bottom[1])
        case 0:
          for(int i = 0; i < num_track_; i++) {
            for (int j = 0; j < num_seg_; j++) {
              *(top_data) = (i < num_seg_) ?
              overlaps_data[n * overlaps_offset_ + i * num_seg_ + j]: 0;
              top_data++;
            }
          }
          break;
          //forward bottom[0]
        case 1:
          caffe_copy(input_offset_, input_data + input_offset_ * n, top_data);
          top_data += input_offset_;
          break;
        default:
          LOG(FATAL) << "select_data can be whether 0 or 1";
      }
    }
  }
  
  template <typename Dtype>
  void SelectLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    //CHECK(!propagate_down[2]) << "Can not propagate to select gate!";
    
    const Dtype* select_data = bottom[2]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    
    for (int n = 0; n < N_; ++n) {
      switch((int)select_data[n]) {
        //set output to the predifined value
        case 0:
          caffe_set(input_offset_, (Dtype).0, bottom_diff + n * input_offset_);	
          break;
          //forward bottom[0]
        case 1:
          caffe_copy(input_offset_, top_diff + input_offset_ * n, bottom_diff + input_offset_ * n);
          break;
        default:
          LOG(FATAL) << "select_data can be whether 0 or 1";
      }
    }
  }
  
  
  #ifdef CPU_ONLY
  STUB_GPU(SelectLayer);
  #endif
  
  INSTANTIATE_CLASS(SelectLayer);
  REGISTER_LAYER_CLASS(Select);
}  // namespace caffe
