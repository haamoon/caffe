#include <vector>

#include "caffe/blob.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/tracker_math.hpp"
#include "caffe/tracking_layers.hpp"

namespace caffe {

template <typename Dtype>
void SegmentPoolingLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  const Dtype* input = bottom[0]->gpu_data();
  
  const Dtype* pooling_data = bottom[1]->gpu_data();
  const Dtype* pooling_indices = bottom[2]->gpu_data();
  const Dtype* pooling_ptr = bottom[3]->gpu_data();
  const Dtype* seg_num = bottom[4]->cpu_data();
  
  const Dtype* pooling_ptr_cpu = bottom[3]->cpu_data();
  
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_gpu_set<Dtype>(top[0]->count(), (Dtype) 0.0, top_data);
  
  for(int n = 0; n < N_; n++) {
    int n_rows = seg_num[n] * nspatial_cell_;
    if(n_rows > 0) {
      CHECK_LE(n_rows, max_nrows_);
      CHECK_EQ(n_rows / nspatial_cell_, (float) n_rows / nspatial_cell_);
      int nnz = pooling_ptr_cpu[n_rows];
      tracker_gpu_csr_gemm<Dtype>(CblasNoTrans, CblasTrans, n_rows,
                                  channels_,
                                  input_ncell_, (Dtype) 1., nnz, pooling_data,
                                  pooling_indices, pooling_ptr, input,
                                  (Dtype) 0.,
                                  top_data, CblasRowMajor); 
    }
    
    input += channels_ * input_ncell_;
    pooling_data += max_data_len_;
    pooling_indices += max_data_len_;
    pooling_ptr += max_nrows_ + 1;
    top_data += max_nrows_ * channels_;
  }
}

template <typename Dtype>
void SegmentPoolingLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  
  const Dtype* pooling_data = bottom[1]->gpu_data();
  const Dtype* pooling_indices = bottom[2]->gpu_data();
  const Dtype* pooling_ptr = bottom[3]->gpu_data();
  const Dtype* pooling_ptr_cpu = bottom[3]->cpu_data();
  const Dtype* seg_num = bottom[4]->cpu_data();
  
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* top_diff = top[0]->gpu_diff();
  
  caffe_gpu_set<Dtype>(bottom[0]->count(), (Dtype) 0.0, bottom_diff);
  for(int n = 0; n < N_; n++) {
    int n_rows = seg_num[n] * nspatial_cell_;
    if(n_rows > 0) {
      int nnz = pooling_ptr_cpu[n_rows];
      tracker_gpu_csr_gemm<Dtype>(CblasTrans, CblasNoTrans, input_ncell_,
                                channels_,
                                n_rows, (Dtype) 1., nnz, pooling_data,
                                pooling_indices, pooling_ptr, top_diff,
                                (Dtype) 0.,
                                bottom_diff,
                                CblasColMajor);
    }
    bottom_diff += channels_ * input_ncell_;
    pooling_data += max_data_len_;
    pooling_indices += max_data_len_;
    pooling_ptr += max_nrows_ + 1;
    top_diff += max_nrows_ * channels_;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SegmentPoolingLayer);

}  // namespace caffe
