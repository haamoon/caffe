#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/tracking_layers.hpp"

namespace caffe {

template <typename Dtype>
void HotMultLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {    
}

template <typename Dtype>
void HotMultLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  HotMultParameter hotmult_param = this->layer_param_.hotmult_param();
  const string mode = hotmult_param .mode();
  if(mode.compare("ROW") == 0) {
    row_mode_ = true;
  } else if(mode.compare("COLUMN") == 0) {
    row_mode_ = false;
  } else {
    LOG(FATAL) << "Undefined hot mode " << mode;
  }

  int a_start_axis = 0;
  int b_start_axis = 0;
  
  a_shape_ = bottom[0]->shape();
  b_shape_ = bottom[1]->shape();
  CHECK_GE(a_shape_.size(), 1);
  a_start_axis = a_shape_.size() - 1;
  a_len_ = a_shape_[a_start_axis];
  
  CHECK_GE(b_shape_.size(), 2);
  b_start_axis = b_shape_.size() - 2;
  b_r_ = b_shape_[b_start_axis];
  b_c_ = b_shape_[b_start_axis + 1];

  N_ = bottom[0]->count(0, a_start_axis);
  CHECK_EQ(N_, bottom[1]->count(0, b_start_axis)) << "Num of Matrices should be the same";

  c_shape_.clear();
  c_shape_.insert(c_shape_.end(), a_shape_.begin(), a_shape_.begin() + a_start_axis); 
  c_shape_.push_back(row_mode_? a_len_: b_r_);
  c_shape_.push_back(row_mode_? b_c_: a_len_);
  top[0]->Reshape(c_shape_); 
}


//permute rows of matrix 'b' with respect to the values in array 'a'.  if from_ind_to_val == True: moves row i to a[i] and row a[i] to i otherwise
// Let matrix A be the one-hot represenation of array 'a' (expanded in column direction) then:
// if from_ind_to_value: C = A^\top x B  else: C = A x B
template <typename Dtype>
void HotMultLayer<Dtype>::permute_row(int N, const Dtype* a, const Dtype* a_lens, int a_len, const Dtype* b, int b_row, int b_col, Dtype* c, int c_row, int c_col, bool from_ind_to_value) {
  CHECK_EQ(b_col, c_col);
  caffe_set(N * c_row * c_col, Dtype(0.0), c);
  for(int n = 0; n < N_; n++) {
    int cur_a_len = (a_lens == NULL) ? a_len : a_lens[n];
    for(int i = 0; i < cur_a_len; i++) {
      int from = from_ind_to_value ? i : a[i];
      int to = from_ind_to_value ? a[i] : i;
      CHECK_LT(from, b_row);
      CHECK_LT(to, c_row);
      int cur_to = to * c_col;
      for(int cur_from = from * b_col; cur_from < (from * b_col) + b_col; cur_from++, cur_to++) {
        c[cur_to] += b[cur_from];
      }
    }
    a += a_len;
    b += b_row * b_col;
    c += c_row * c_col;
  }
}

//permute columns of matrix 'b' with respect to the values in array 'a'.  If from_ind_to_val == True: moves column i to a[i] and moves column a[i] to i otherwise
// Let matrix A be the one-hot represenation of array 'a' (expanded in row direction) then:
// if(from_ind_to_value): C = B x A^\top else: C = B x A 
// c_col is needed only in (from_ind_to_value == true) case
template <typename Dtype>
void HotMultLayer<Dtype>::permute_col(int N, const Dtype* a, const Dtype* a_lens, int a_len, const Dtype* b, int b_row, int b_col, Dtype* c, int c_row, int c_col, bool from_ind_to_value) {
  CHECK_EQ(b_row, c_row);
  caffe_set(N * c_row * c_col, Dtype(0.0), c);
  for(int n = 0; n < N; n++) {
    int cur_a_len = (a_lens == NULL) ? a_len : a_lens[n];
    for(int i = 0; i < cur_a_len; i++) {
      int from = from_ind_to_value ? i : a[i];
      int to = from_ind_to_value ? a[i] : i;
      
      CHECK_LT(from, b_col);
      CHECK_LT(to, c_col);
      
      int to_offset = 0;
      for(int from_offset = 0; from_offset < b_row * b_col; from_offset += b_col, to_offset += c_col) {
        c[to_offset + to] += b[from_offset + from];
      }
    }
    a += a_len;
    b += b_row * b_col;
    c += c_row * c_col;
  }
}

template <typename Dtype>
void HotMultLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  
  const Dtype* A_data = bottom[0]->cpu_data();
  const Dtype* B_data = bottom[1]->cpu_data();
  const Dtype* a_lens = NULL; 
  if(bottom.size() > 2) {
    a_lens = bottom[2]->cpu_data();
  }
  Dtype* C_data = top[0]->mutable_cpu_data();
  
  if(row_mode_) {
    //C = A x B
    this->permute_row(N_, A_data, a_lens, a_len_, B_data, b_r_, b_c_, C_data, a_len_, b_c_, false);
    //LOG(FATAL);
  } else {
    //C = B x A
    this->permute_col(N_, A_data, a_lens, a_len_, B_data, b_r_, b_c_, C_data, b_r_, a_len_, false);
  }
}

template <typename Dtype>
void HotMultLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* A_data = bottom[0]->cpu_data();
  const Dtype* C_diff = top[0]->cpu_diff();
  Dtype* B_diff = bottom[1]->mutable_cpu_diff();
  
  const Dtype* a_lens = NULL; 
  if(bottom.size() > 2) {
    a_lens = bottom[2]->cpu_data();
  }
  
  
  if(row_mode_) {
    //B' = A^\top x C'
    this->permute_row(N_, A_data, a_lens, a_len_, C_diff, a_len_, b_c_, B_diff, b_r_, b_c_, true);
  } 
  else {
    //B' = C' x A^\top
    this->permute_col(N_, A_data, a_lens, a_len_, C_diff, b_r_, a_len_, B_diff, b_r_, b_c_, true);
  }
}

#ifdef CPU_ONLY
STUB_GPU(HotMultLayer);
#endif


INSTANTIATE_CLASS(HotMultLayer);
REGISTER_LAYER_CLASS(HotMult);

}  // namespace caffe
