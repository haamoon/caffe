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
void RowPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void RowPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  
  CHECK_GE(bottom[0]->num_axes(), 3) << "Input(0) image must have at least 3 axes, "
      << "corresponding to (..., row, col)";
  CHECK_GE(bottom[1]->num_axes(), 2) << "Input(1) seg_data must have at least 2 axes, "
      << "corresponding to (..., segment_inds)";
  CHECK_GE(bottom[2]->num_axes(), 2) << "Input(2) seg_ptr must have at least 2 axes, "
      << "corresponding to (..., segment_start_inds)";
  CHECK_GE(bottom[3]->num_axes(), 1) << "Input(3) seg_num must have at least 1 axes, "
      << "corresponding to (...)";
  CHECK_GE(bottom[4]->num_axes(), 2) << "Input(4) seg_coef must have at least 2 axes, "
      << "corresponding to (..., seg_coef)";
  
  int input_start_axis = bottom[0]->CanonicalAxisIndex(-2);
  N_ = bottom[0]->count(0, input_start_axis);
  nrow_ = bottom[0]->shape(input_start_axis);
  ncol_ = bottom[0]->shape(input_start_axis + 1);  
  
  //seg_data is a ... x seg_data_len_ matrix
  seg_data_len_ = bottom[1]->shape(bottom[1]->CanonicalAxisIndex(-1)); 
  
  //seg_ptr is a ... x seg_ptr_len_ matrix
  seg_ptr_len_ = bottom[2]->shape(bottom[2]->CanonicalAxisIndex(-1));
  
  CHECK_EQ(N_, bottom[1]->count(0, bottom[1]->CanonicalAxisIndex(-1)));
  CHECK_EQ(N_, bottom[2]->count(0, bottom[2]->CanonicalAxisIndex(-1)));
  CHECK_EQ(N_, bottom[3]->count(0, bottom[3]->CanonicalAxisIndex(-1) + 1));
  CHECK_EQ(N_, bottom[4]->count(0, bottom[4]->CanonicalAxisIndex(-1)));
  CHECK_EQ(seg_data_len_, bottom[4]->shape(bottom[1]->CanonicalAxisIndex(-1)));
  
  //num of segments is (seg_ptr_len_ - 1)
  //X_t = top is a (seg_ptr_len_ - 1) x ncol_ matrix
  vector<int> top_shape = bottom[3]->shape();
  top_shape.push_back(seg_ptr_len_ - 1);
  top_shape.push_back(ncol_);
  
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void RowPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const Dtype* matrix_data = bottom[0]->cpu_data();
  const Dtype* seg_data = bottom[1]->cpu_data();
  const Dtype* seg_ptr = bottom[2]->cpu_data();
  const Dtype* seg_num = bottom[3]->cpu_data();
  const Dtype* seg_coef = bottom[4]->cpu_data();
  
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
 
  for (int i = 0; i < top_count; ++i) {
    top_data[i] = 0;
  }
  
  // The main loop
  for (int n = 0; n < N_; ++n) {
    //iterate over segments
    //LOG(ERROR) << "seg_num[n] " << seg_num[n];
    for(int seg = 0; seg < seg_num[n]; ++seg) {
      int start_ind = seg_ptr[seg];
      int end_ind = seg_ptr[seg + 1];
      //LOG(ERROR) << "n = " << n << " start, end = " << start_ind << ", " << end_ind;
      for (int col = 0; col < ncol_; ++col) {
        for(int i = start_ind; i < end_ind; i++) {
          top_data[seg * ncol_ + col] += matrix_data[(int)(seg_data[i]) * ncol_ + col] * seg_coef[i];
        }
      }
    }
    matrix_data += bottom[0]->count(bottom[0]->CanonicalAxisIndex(-2));
    seg_coef += seg_data_len_;
    top_data += ncol_ * (seg_ptr_len_ - 1);
    seg_ptr += seg_ptr_len_;
    seg_data += seg_data_len_;
  }
}

template <typename Dtype>
void RowPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  CHECK(!propagate_down[1]) << "Can not backpropagate to seg_data";
  CHECK(!propagate_down[2]) << "Can not backpropagate to seg_ptr";
  CHECK(!propagate_down[3]) << "Can not backpropagate to seg_num";
  CHECK(!propagate_down[4]) << "Can not backpropagate to seg_coef";

  if (!propagate_down[0]) {
    return;
  }
    
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* seg_data = bottom[1]->cpu_data();
  const Dtype* seg_ptr = bottom[2]->cpu_data();
  const Dtype* seg_num = bottom[3]->cpu_data();
  const Dtype* seg_coef = bottom[4]->cpu_data();


  for (int i = 0; i < bottom[0]->count(); ++i) {
    bottom_diff[i] = 0;
  }
  
  // The main loop
  for (int n = 0; n < N_; ++n) {
    //iterate over segments
    for(int seg = 0; seg < seg_num[n]; ++seg) {
      int start_ind = seg_ptr[seg]; 
      int end_ind = seg_ptr[seg + 1];
      for (int col = 0; col < ncol_; ++col) {
        for(int i = start_ind; i < end_ind; i++) {
          bottom_diff[(int)(seg_data[i]) * ncol_ + col] += top_diff[seg * ncol_ + col] * seg_coef[i];
        }
      }
    }
    bottom_diff += bottom[0]->count(bottom[0]->CanonicalAxisIndex(-2));
    seg_coef += seg_data_len_;
    top_diff += ncol_ * (seg_ptr_len_ - 1);	
    seg_ptr += seg_ptr_len_;
    seg_data += seg_data_len_;
  }
}


#ifdef CPU_ONLY
STUB_GPU(RowPoolingLayer);
#endif

INSTANTIATE_CLASS(RowPoolingLayer);
REGISTER_LAYER_CLASS(RowPooling);

}  // namespace caffe
