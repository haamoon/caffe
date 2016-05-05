#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/tracking_layers.hpp"
#include "caffe/util/tracker_math.hpp"
namespace caffe {

template <typename Dtype>
void Filter2DLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) { 
  //mode_ =? "filter" : "set_zero"
  Filter2DParameter filter_param = this->layer_param_.filter2d_param();
  mode_ = filter_param.mode();
}

template <typename Dtype>
void Filter2DLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // bottom[0] are d1xd2...xdn blobs to mask
  // bottom[1] is the d1xd2x...d{n - 2}  row mask
  // optional
  // bottom[2] is the d1xd2x...d{n - 2}  col mask
 
  if(bottom.size() > 2) {
    CHECK(bottom[1]->shape() == bottom[2]->shape());
  }
  
  input_rows_ = bottom[0]->shape(-2);
  input_cols_ = bottom[0]->shape(-1);
  
  CHECK_EQ(bottom[0]->num_axes() - 2, bottom[1]->num_axes());
  
  for (int i = 0; i < (bottom[1]->num_axes() - 1); ++i) {
      CHECK_EQ(bottom[1]->shape(i), bottom[0]->shape(i));
  }  
  
  if(mode_.compare("filter") == 0) {
    const Dtype* n_rows = bottom[1]->cpu_data();
    const Dtype* n_cols = NULL;
    if(bottom.size() > 2) {
      n_cols = bottom[2]->cpu_data();
    }
    
    int size = 0;
    int cols_i = input_cols_;
    for(int i = 0; i < bottom[1]->count(); ++i) {
      if(n_cols != NULL) {
        cols_i = n_cols[i];
        CHECK_GE(cols_i, 0);
        CHECK_LE(cols_i, input_cols_);
        CHECK_EQ(cols_i, static_cast<int>(cols_i));
      }
      size += n_rows[i] * cols_i;
      
      CHECK_GE(n_rows[i], 0);
      CHECK_LE(n_rows[i], input_rows_);
      CHECK_EQ(n_rows[i], static_cast<int>(n_rows[i]));
    }
    
    vector<int> top_shape;
    top_shape.push_back(size == 0 ? 1 : size);
    top[0]->Reshape(top_shape);
  } else if(mode_.compare("set_zero") == 0) {
    top[0]->ReshapeLike(*bottom[0]);
    
  } else {
    LOG(FATAL) << "Unknown mode " << mode_;
  }
  
 
}

template <typename Dtype>
void Filter2DLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* n_rows = bottom[1]->cpu_data();
  const Dtype* n_cols = NULL;
  if(bottom.size() > 2) {
    n_cols = bottom[2]->cpu_data();
  } 
  Dtype* top_data = top[0]->mutable_cpu_data();
  top_data[0] = 0;
  
  int input_offset = 0;
  int n_cols_i = input_cols_;
  
  if(mode_.compare("filter") == 0) {
    int filter_index = 0;
    for(int i = 0; i < bottom[1]->count(); ++i) {
      if(n_cols != NULL) {
        n_cols_i = n_cols[i];
      }
      
      for(int r = 0; r < n_rows[i]; r++) {
        
        for(int c = 0; c < n_cols_i; c++) {
          int input_index = input_offset + r * input_cols_ + c;
          top_data[filter_index] = input_data[input_index];
          filter_index++;
        }
      }
      input_offset += input_rows_ * input_cols_;
    }  
  } else {
    int offset = 0;
    caffe_set(top[0]->count(), Dtype(0.0), top_data);
    for(int i = 0; i < bottom[1]->count(); ++i) {
      CHECK_LE(n_rows[i], input_rows_);
      CHECK_EQ(n_rows[i], static_cast<int>(n_rows[i]));
      CHECK_GE(n_rows[i], 0);
      if(n_cols != NULL) {
        n_cols_i = n_cols[i];
        CHECK_LE(n_cols_i, input_cols_);
        CHECK_EQ(n_cols_i, static_cast<int>(n_cols_i));
        CHECK_GE(n_cols_i, 0);
      }
      
      for(int r = 0; r < n_rows[i]; r++) {
        for(int c = 0; c < n_cols_i; c++) {
          top_data[r * input_cols_ + c] = input_data[offset + r * input_cols_ + c];
        }
      }
      top_data += input_rows_ * input_cols_;
      input_data += input_rows_ * input_cols_;
    }
  }
}

template <typename Dtype>
void Filter2DLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  Dtype* input_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* n_rows = bottom[1]->cpu_data();
  const Dtype* n_cols = NULL;
  if(bottom.size() > 2) {
    n_cols = bottom[2]->cpu_data();
  } 
  const Dtype* top_diff = top[0]->cpu_diff();

  int offset = 0;
  caffe_set(bottom[0]->count(), Dtype(0.0), input_diff);
  int n_cols_i = input_cols_;
  
  if(mode_.compare("filter") == 0) {
    int filter_index = 0;
    for(int i = 0; i < bottom[1]->count(); ++i) {
      
      for(int r = 0; r < n_rows[i]; r++) {      
        if(n_cols != NULL) {
          n_cols_i = n_cols[i];
        }
        for(int c = 0; c < n_cols_i; c++) {
          int input_index = offset + r * input_cols_ + c;
          input_diff[input_index] = top_diff[filter_index];
          filter_index++;
        }
      }
      offset += input_rows_ * input_cols_;
    }  
  } else {
    for(int i = 0; i < bottom[1]->count(); ++i) {
      for(int r = 0; r < n_rows[i]; r++) {
        if(n_cols != NULL) {
          n_cols_i = n_cols[i];
        }
        for(int c = 0; c < n_cols_i; c++) {
          input_diff[r * input_cols_ + c] = top_diff[r * input_cols_ + c];
        }
      }
      top_diff += input_rows_ * input_cols_;
      input_diff += input_rows_ * input_cols_;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(Filter2DLayer);
#endif

INSTANTIATE_CLASS(Filter2DLayer);
REGISTER_LAYER_CLASS(Filter2D);

}  // namespace caffe
