#include <vector>

#include "caffe/tracking_layers.hpp"

namespace caffe {

template <typename Dtype>
void SwitchLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param_.switch_param().axis());
  
  vector<int> top_shape = bottom[0]->shape();
  
  // Check that the dimensions of bottoms are all the same
  for (int i = 1; i < bottom.size() - 1; ++i) {
    vector<int> shape_i = bottom[i]->shape();
    CHECK(shape_i == top_shape);
  }
  
  // Check the selector dimensions
  // It could be generalized to have more channels, one per top
  const int selector_ind = bottom.size() - 1;
  
  CHECK_EQ(axis_, bottom[selector_ind]->num_axes())
  << "selector axis should be equal to the staring axis " << axis_;
  
  for (int i = 0; i < bottom[selector_ind]->num_axes(); ++i) {
    CHECK_EQ(bottom[0]->shape(i), bottom[selector_ind]->shape(i))
    << "dimension mismatch between bottom[0]->shape(" << i
    << ") and bottom["<< selector_ind << "]->shape(" << i << ")";
  }
  outer_dim_ = bottom[0]->count(0, axis_);
  selector_dim_ = bottom[selector_ind]->count();
  inner_dim_ = bottom[0]->count(axis_);
}

template <typename Dtype>
void SwitchLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Initialize with the first blob.
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void SwitchLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int selector_ind = bottom.size() - 1;
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* select_data = bottom[selector_ind]->cpu_data();
  
  for (int n = 0; n < selector_dim_; n++) {
    int index = static_cast<int>(select_data[n]);
    DCHECK(floor(index) == index) << "Index should be an integer";
    DCHECK_GE(index, 0) << "Index should be greater than 0";
    DCHECK_LT(index, selector_ind) << "Index should be less than " << selector_ind;
    const Dtype* bottom_data = bottom[index]->cpu_data();
    caffe_copy(inner_dim_, bottom_data + n * inner_dim_,
               top_data + n * inner_dim_);
  }
}

template <typename Dtype>
void SwitchLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int selector_ind = bottom.size() - 1;
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* select_data = bottom[selector_ind]->cpu_data();
  //CHECK(!propagate_down[selector_ind]) << "Switch layer cannot backpropagate to selector inputs.";

  for (int n = 0; n < selector_dim_; n++) {
    int index = static_cast<int>(select_data[n]);
    Dtype* bottom_diff = bottom[index]->mutable_cpu_diff();
    caffe_copy(inner_dim_, top_diff+ n * inner_dim_,
               bottom_diff + n * inner_dim_);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SwitchLayer);
#endif

INSTANTIATE_CLASS(SwitchLayer);
REGISTER_LAYER_CLASS(Switch);
}  // namespace caffe
