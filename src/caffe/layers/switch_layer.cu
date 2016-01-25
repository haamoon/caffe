#include <vector>

#include "caffe/layer.hpp"
#include "caffe/tracking_layers.hpp"

namespace caffe {


template <typename Dtype>
void SwitchLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int selector_ind = bottom.size() - 1;
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* select_data = bottom[selector_ind]->cpu_data();
  
  for (int n = 0; n < selector_dim_; n++) {
    int index = static_cast<int>(select_data[n]);
    DCHECK(floor(index) == index) << "Index should be an integer";
    DCHECK_GE(index, 0) << "Index should be greater than 0";
    DCHECK_LT(index, selector_ind) << "Index should be less than " << selector_ind;
    const Dtype* bottom_data = bottom[index]->gpu_data();
    caffe_copy(inner_dim_, bottom_data + n * inner_dim_,
               top_data + n * inner_dim_);
  }
}

template <typename Dtype>
void SwitchLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  const int selector_ind = bottom.size() - 1;
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* select_data = bottom[selector_ind]->cpu_data();
  
  //CHECK(!propagate_down[selector_ind]) << " Switch layer cannot backpropagate to selector inputs.";
  
  for (int n = 0; n < selector_dim_; n++) {
    int index = static_cast<int>(select_data[n]);
    Dtype* bottom_diff = bottom[index]->mutable_gpu_diff();
    caffe_copy(inner_dim_, top_diff+ n * inner_dim_,
               bottom_diff + n * inner_dim_);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SwitchLayer);

}  // namespace caffe
