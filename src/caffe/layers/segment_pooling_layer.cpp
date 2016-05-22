#include <vector>

#include "caffe/blob.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/tracker_math.hpp"
#include "caffe/tracking_layers.hpp"

namespace caffe {
  //This layer compute C = P * F'
  //where P is a (...x(num_segs x nspatial_cell_)xinput_ncell_) sparse matrix which is represented in csr format 
  //==> data (...x max_data_len_): bottom[1], indices (...x max_data_len_) : bottom[2], ptr (...x (max_nrows_ + 1)): bottom[3]
  //F is a dense (...x channels_ x H x W) where H x W = input_ncell_
  //result will be a (...x(num_segs x nspatial_cell_)xchannels_) matrix 
  //then, it is reshaped into (...x(num_segs)x(nspatial_cell_ x channels_)) segment feature matrix
  //seg_num: bottom[4]
  template <typename Dtype>
  void SegmentPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                              const vector<Blob<Dtype>*>& top) {
    
    //For each segment we have nspatial_cell_ rows in the sparse matrix
    nspatial_cell_ = this->layer_param_.segment_pooling_param().num_spatial_cells();
    
    int input_start_axis = bottom[0]->CanonicalAxisIndex(-3);
    N_ = bottom[0]->count(0, input_start_axis);
    
    CHECK_EQ(N_, bottom[1]->count(0, bottom[1]->CanonicalAxisIndex(-1)));
    CHECK_EQ(N_, bottom[2]->count(0, bottom[2]->CanonicalAxisIndex(-1)));
    CHECK_EQ(N_, bottom[3]->count(0, bottom[3]->CanonicalAxisIndex(-1)));
    CHECK_EQ(N_, bottom[4]->count());
    
    channels_ = bottom[0]->shape(input_start_axis);
    input_ncell_ = bottom[0]->count(input_start_axis + 1);
    
    max_data_len_ = bottom[1]->count(bottom[1]->CanonicalAxisIndex(-1));
    CHECK_EQ(max_data_len_, bottom[2]->count(bottom[1]->CanonicalAxisIndex(-1)));
    max_nrows_ = bottom[3]->count(bottom[3]->CanonicalAxisIndex(-1)) - 1;
    
    CHECK_EQ(max_nrows_ / nspatial_cell_, (float) max_nrows_ / nspatial_cell_);
    
    
    int_indices_.Reshape(bottom[2]->shape());
    int_ptrs_.Reshape(bottom[3]->shape());
    
    vector<int> top_shape = bottom[4]->shape();
    
    // for N_ == 1 we can use actual seg_num value
    //if(N_ == 1 && *(bottom[4]->cpu_data()) > 0) {
    //  int seg_num  = *(bottom[4]->cpu_data());
    //  top_shape.push_back(seg_num);
    //} else {
    top_shape.push_back(max_nrows_/nspatial_cell_);
    //}
    
    top_shape.push_back(channels_ * nspatial_cell_);
    
    
    
    top[0]->Reshape(top_shape);
}
                                              
template <typename Dtype>
void SegmentPoolingLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* input = bottom[0]->cpu_data();
  const Dtype* pooling_data = bottom[1]->cpu_data();
  const Dtype* seg_num = bottom[4]->cpu_data();
  
  
  int* int_indices_data = int_indices_.mutable_cpu_data();
  int* int_ptrs_data = int_ptrs_.mutable_cpu_data();
  
  tracker_cpu_toInt(bottom[2]->count(), bottom[2]->cpu_data(), int_indices_data);
  tracker_cpu_toInt(bottom[3]->count(), bottom[3]->cpu_data(), int_ptrs_data);

  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set<Dtype>(top[0]->count(), (Dtype) 0.0, top_data);
  for(int n = 0; n < N_; n++) {
    int n_rows = seg_num[n] * nspatial_cell_;
    CHECK_LE(n_rows, max_nrows_);
    CHECK_EQ(n_rows / nspatial_cell_, (float) n_rows / nspatial_cell_);
    int nnz = int_ptrs_data[n_rows + 1];
    tracker_cpu_csr_gemm<Dtype>(CblasNoTrans, CblasTrans, n_rows,
                              channels_,
                              input_ncell_, (Dtype) 1., nnz, pooling_data,
                              int_indices_data, int_ptrs_data, input,
                              (Dtype) 0.,
                              top_data, CblasRowMajor);
    
    
    input += channels_ * input_ncell_;
    pooling_data += max_data_len_;
    int_indices_data += max_data_len_;
    int_ptrs_data += max_nrows_ + 1;
    top_data += max_nrows_ * channels_;
  }
}

template <typename Dtype>
void SegmentPoolingLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* pooling_data = bottom[1]->cpu_data();
  const int* int_indices_data = int_indices_.cpu_data();
  const int* int_ptrs_data = int_ptrs_.cpu_data();
  const Dtype* seg_num = bottom[4]->cpu_data();
  
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* top_diff = top[0]->cpu_diff();
  
  
  caffe_set<Dtype>(bottom[0]->count(), (Dtype) 0.0, bottom_diff);
  for(int n = 0; n < N_; n++) {
    int n_rows = seg_num[n] * nspatial_cell_;
    int nnz = int_ptrs_data[n_rows + 1];
    tracker_cpu_csr_gemm<Dtype>(CblasTrans, CblasNoTrans, input_ncell_,
                              channels_,
                              n_rows, (Dtype) 1., nnz, pooling_data,
                              int_indices_data, int_ptrs_data, top_diff,
                              (Dtype) 0.,
                              bottom_diff,
                              CblasColMajor);
  
    bottom_diff += channels_ * input_ncell_;
    pooling_data += max_data_len_;
    int_indices_data += max_data_len_;
    int_ptrs_data += max_nrows_ + 1;
    top_diff += max_nrows_ * channels_;
  }
}

#ifdef CPU_ONLY
STUB_GPU(SegmentPoolingLayer);
#endif

INSTANTIATE_CLASS(SegmentPoolingLayer);
REGISTER_LAYER_CLASS(SegmentPooling);

}  // namespace caffe


