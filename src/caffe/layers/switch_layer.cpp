#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/tracking_layers.hpp"

namespace caffe {
  
  template <typename Dtype>
  void SwitchLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top) {
    
    vector<int> input_shape = bottom[0]->shape();
    vector<int> switch_shape = bottom[1]->shape();
    
    CHECK_GE(input_shape.size(), 2);
    int input_start_axis = input_shape.size() - 2;
    
    D_1_ = input_shape[input_start_axis];
    D_2_ = input_shape[input_start_axis + 1];
    input_offset_ = D_1_ * D_2_;
    
    N_ = bottom[0]->count(0, input_start_axis);
    CHECK_EQ(N_, bottom[1]->count());
    
    
    //Reshaping top
    top[0]->ReshapeLike(*bottom[0]);
  }
  
  template <typename Dtype>
  void SwitchLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
    
    const Dtype* input_data = bottom[0]->cpu_data();
    const Dtype* switch_data = bottom[1]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    
    
    for (int n = 0; n < N_; ++n) {
      switch((int)switch_data[n]) {
          //set output to the predifined value
        case 0:
          caffe_set(input_offset_, (Dtype).0, top_data + n * input_offset_);
          caffe_strided_add_scalar<Dtype>(input_offset_, (Dtype)1.0, D_2_ + 1, top_data + input_offset_ * n);
          break;
          //forward bottom[0]
        case 1:
          caffe_copy(input_offset_, input_data + input_offset_ * n, top_data + input_offset_ * n);
          break;
        default:
          LOG(FATAL) << "switch_data can be whether 0 or 1";
      }
    }
  }
  
  template <typename Dtype>
  void SwitchLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<Dtype>*>& bottom) {
    //CHECK(!propagate_down[1]) << "Can not propagate to switch gate!";
    
    const Dtype* switch_data = bottom[1]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    
    for (int n = 0; n < N_; ++n) {
      switch((int)switch_data[n]) {
          //set output to the predifined value
        case 0:
          caffe_set(input_offset_, (Dtype).0, bottom_diff + n * input_offset_);
          break;
          //forward bottom[0]
        case 1:
          caffe_copy(input_offset_, top_diff + input_offset_ * n, bottom_diff + input_offset_ * n);
          break;
        default:
          LOG(FATAL) << "switch_data can be whether 0 or 1";
      }
    }
  }
  
  
#ifdef CPU_ONLY
  STUB_GPU(SwitchLayer);
#endif
  
  INSTANTIATE_CLASS(SwitchLayer);
  REGISTER_LAYER_CLASS(Switch);
}  // namespace caffe
