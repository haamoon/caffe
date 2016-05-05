#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/tracker_math.hpp"
#include "caffe/tracking_layers.hpp"

namespace caffe {

template <typename Dtype>
void Filter2DLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* input_data = bottom[0]->gpu_data();
  const Dtype* n_rows = bottom[1]->cpu_data();
  const Dtype* n_cols = NULL;
  if(bottom.size() > 2) {
    n_cols = bottom[2]->cpu_data();
  } 
  Dtype* top_data = top[0]->mutable_gpu_data();
  caffe_gpu_set(1, Dtype(0.0), top_data);
  
  
  int input_offset = 0;
  
  if(mode_.compare("filter") == 0) {
    int filter_offset = 0;
    for(int i = 0; i < bottom[1]->count(); ++i) {  
      if(n_cols == NULL) {
        copy_gpu_matrix<Dtype>(n_rows[i], input_cols_, input_data + input_offset, input_cols_, top_data + filter_offset, input_cols_);
        filter_offset += n_rows[i] * input_cols_;
      } else {
        copy_gpu_matrix<Dtype>(n_rows[i], n_cols[i], input_data + input_offset, input_cols_, top_data + filter_offset, n_cols[i]);
        filter_offset += n_rows[i] * n_cols[i];
      }
      input_offset += input_rows_ * input_cols_;
    }
  } else {
    caffe_gpu_set(top[0]->count(), Dtype(0.0), top_data);
    for(int i = 0; i < bottom[1]->count(); ++i) {  
      if(n_cols == NULL) {
        //LOG(ERROR) << "HERE2: "<< n_rows[i];
        CHECK_LE(n_rows[i], input_rows_);
        CHECK_EQ(n_rows[i], static_cast<int>(n_rows[i]));
        CHECK_GE(n_rows[i], 0);
        copy_gpu_matrix<Dtype>(n_rows[i], input_cols_, input_data + input_offset, input_cols_, top_data + input_offset, input_cols_);
      } else {
        //LOG(ERROR) << "HERE: "<< n_rows[i] << ", " << n_cols[i];
        CHECK_LE(n_rows[i], input_rows_);
        CHECK_GE(n_rows[i], 0);
        CHECK_EQ(n_rows[i], static_cast<int>(n_rows[i]));
        CHECK_LE(n_cols[i], input_cols_);
        CHECK_GE(n_cols[i], 0);
        CHECK_EQ(n_cols[i], static_cast<int>(n_cols[i]));
        copy_gpu_matrix<Dtype>(n_rows[i], n_cols[i], input_data + input_offset, input_cols_, top_data + input_offset, input_cols_);
      }
      input_offset += input_rows_ * input_cols_;
    }
  }
  
//   std::stringstream buffer;
//   tracker_printMat(buffer, n_rows, bottom[1]->count(), bottom[1]->count());
//   buffer << std::endl;
//   tracker_printMat(buffer, n_cols, bottom[1]->count(), bottom[1]->count());
//   buffer << "INPUTTTTTTTT:" << std::endl;
//   tracker_printMat(buffer, bottom[0]->cpu_data(), 10, top[0]->count());
//   buffer << "OUTPUTTTTTTT:" << std::endl;
//   tracker_printMat(buffer, top[0]->cpu_data(), 10, top[0]->count());
//   
//   LOG(ERROR) << buffer.str();
}

template <typename Dtype>
void Filter2DLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  Dtype* input_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* n_rows = bottom[1]->cpu_data();
  const Dtype* n_cols = NULL;
  if(bottom.size() > 2) {
    n_cols = bottom[2]->cpu_data();
  } 
  const Dtype* top_diff = top[0]->gpu_diff();
  
  int input_offset = 0;
  caffe_gpu_set(bottom[0]->count(), Dtype(0.0), input_diff);
  
  if(mode_.compare("filter") == 0) {
    int filter_offset = 0;
    for(int i = 0; i < bottom[1]->count(); ++i) {  
      if(n_cols == NULL) {
        copy_gpu_matrix<Dtype>(n_rows[i], input_cols_, top_diff + filter_offset, input_cols_, input_diff + input_offset, input_cols_);
        filter_offset += n_rows[i] * input_cols_;
      } else {
        copy_gpu_matrix<Dtype>(n_rows[i], n_cols[i], top_diff + filter_offset, n_cols[i], input_diff + input_offset, input_cols_);
        filter_offset += n_rows[i] * n_cols[i];
      }
      input_offset += input_rows_ * input_cols_;
    }
  } else {
    for(int i = 0; i < bottom[1]->count(); ++i) {  
      if(n_cols == NULL) {
        copy_gpu_matrix<Dtype>(n_rows[i], input_cols_, top_diff + input_offset, input_cols_, input_diff + input_offset, input_cols_);
      } else {
        copy_gpu_matrix<Dtype>(n_rows[i], n_cols[i], top_diff + input_offset, input_cols_, input_diff + input_offset, input_cols_);
      }
      input_offset += input_rows_ * input_cols_;
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(Filter2DLayer);

}  // namespace caffe
