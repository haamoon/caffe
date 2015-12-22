#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/tracking_layers.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class RowPoolingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  RowPoolingLayerTest() {
    input_ = new Blob<Dtype>();
    seg_data_ = new Blob<Dtype>();
    seg_ptr_ = new Blob<Dtype>();
    seg_num_ = new Blob<Dtype>();
    seg_coef_ = new Blob<Dtype>();
    output_ = new Blob<Dtype>();
  
    T_ = 2;
    N_ = 2;
    C_ = 3;
    seg_ptr_len_ = 4;
    seg_data_len_ = 10;
    

    // fill the values
    Caffe::set_random_seed(1701);
    
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    
    // blob_image shape: T_ x N_ x (seg_ptr_len_ - 1) x C_
    vector<int> input_shape;
    input_shape.push_back(T_);
    input_shape.push_back(N_);
    input_shape.push_back(4);
    input_shape.push_back(C_);
    input_->Reshape(input_shape);
    filler.Fill(input_);
    
    //seg_data_shape: T_ x N_ x seg_data_len_
    vector<int> seg_data_shape;
    //seg_data_shape.push_back(T_);
    seg_data_shape.push_back(T_ * N_);
    seg_data_shape.push_back(seg_data_len_);
    seg_data_->Reshape(seg_data_shape);
    seg_coef_->Reshape(seg_data_shape);
    
    Dtype* seg_data_array = seg_data_->mutable_cpu_data();
    Dtype* seg_coef_array = seg_coef_->mutable_cpu_data();
    
    for(int i = 0; i < 2; i++) {
      //mask for input 1, seg 1
      seg_data_array[0] = 0;
      seg_data_array[1] = 1;
      seg_data_array[2] = 2;
      
      if(i == 0) {
        seg_coef_array[0] = 0;
        seg_coef_array[1] = 0;
        seg_coef_array[2] = 0;
      } else {
        seg_coef_array[0] = 1;
        seg_coef_array[1] = 2;
        seg_coef_array[2] = 3;
      }
      
      //mask for input 1, seg 2
      seg_data_array[3] = 0;
      seg_data_array[4] = 2;
      seg_data_array[5] = 1;
      seg_data_array[6] = 3;

      if(i == 0) {
        seg_coef_array[3] = 0;
        seg_coef_array[4] = 0;
        seg_coef_array[5] = 0;
        seg_coef_array[6] = 0;
      } else {
        seg_coef_array[3] = 1;
        seg_coef_array[4] = 2;
        seg_coef_array[5] = 3;
        seg_coef_array[6] = 4;
      }
      
      seg_data_array += seg_data_len_;
      seg_coef_array += seg_data_len_;
      
      //mask for image 2, seg 1
      seg_data_array[0] = 0;
      seg_data_array[1] = 2;
      
      if(i == 0) {
        seg_coef_array[0] = 0;
        seg_coef_array[1] = 0;
      } else {
        seg_coef_array[0] = 0;
        seg_coef_array[1] = 2;
      }
      
      //mask for image 2, seg 2
      seg_data_array[2] = 1;
      seg_data_array[3] = 3;
      
      if(i == 0) {
        seg_coef_array[2] = 0;
        seg_coef_array[3] = 0;
      } else {
        seg_coef_array[2] = -1;
        seg_coef_array[3] = 1;
      }
      
      //mask for image 2, seg 3
      seg_data_array[4] = 0;
      seg_data_array[5] = 1;
      
      if(i == 0) {
        seg_coef_array[4] = 0;
        seg_coef_array[5] = 0;
      } else {
        seg_coef_array[4] = 1;
        seg_coef_array[5] = -1;
      }
      seg_data_array += seg_data_len_;
      seg_coef_array += seg_data_len_;
    }
    
    //seg_ptr_shape: T_ x N_ x seg_ptr_len_
    vector<int> seg_ptr_shape;
    seg_ptr_shape.push_back(T_);
    seg_ptr_shape.push_back(N_);
    seg_ptr_shape.push_back(seg_ptr_len_);
    seg_ptr_->Reshape(seg_ptr_shape);
    
    Dtype* seg_ptr_array = seg_ptr_->mutable_cpu_data();
    
    for(int i = 0; i < 2; i++) {
      //inds for image 1
      seg_ptr_array[0] = 0;
      seg_ptr_array[1] = 3;
      seg_ptr_array[2] = 7;
      
      seg_ptr_array += seg_ptr_len_;
      
      //inds for image 2
      seg_ptr_array[0] = 0;
      seg_ptr_array[1] = 2;
      seg_ptr_array[2] = 4;
      seg_ptr_array[3] = 6;
      
      seg_ptr_array += seg_ptr_len_;
    }
    
    //seg_num_shape: T_ x N_
    vector<int> seg_num_shape;
    seg_num_shape.push_back(T_);
    seg_num_shape.push_back(N_);
    seg_num_->Reshape(seg_num_shape);
    Dtype* seg_num_array = seg_num_->mutable_cpu_data();
    
    seg_num_array[0] = 2;
    seg_num_array[1] = 3;
    seg_num_array[2] = 2;
    seg_num_array[3] = 3;
    
    //Add blobs to input vector
    blob_bottom_vec_.push_back(input_);
    blob_bottom_vec_.push_back(seg_data_);
    blob_bottom_vec_.push_back(seg_ptr_);
    blob_bottom_vec_.push_back(seg_num_);
    blob_bottom_vec_.push_back(seg_coef_);
    
    //Add top blob to the output vector
    blob_top_vec_.push_back(output_);
  }
  
  virtual ~RowPoolingLayerTest() {
    delete input_;
    delete seg_data_;
    delete seg_ptr_;
    delete seg_num_;
    delete seg_coef_;
  }
  
  template <typename Dtype>
  void printMat(std::stringstream& buffer, Dtype* mat, int col, int count) {
    for (int i = 0; i < count; ++i) {
      if(i % col == 0) {
        buffer << ';' << endl;
      }
      buffer << mat[i] << ' ';    
      }
    buffer << endl;
  }
  
  Blob<Dtype>* input_;
  Blob<Dtype>* seg_data_;
  Blob<Dtype>* seg_ptr_;
  Blob<Dtype>* seg_num_;
  Blob<Dtype>* seg_coef_;
  Blob<Dtype>* output_;
  int T_;
  int N_;
  int C_;
  int seg_ptr_len_;
  int seg_data_len_;
  
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(RowPoolingLayerTest, TestDtypesAndDevices);

TYPED_TEST(RowPoolingLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  RowPoolingLayer<Dtype>layer(layer_param);
      
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  
  //T_ x N_ x (seg_ptr_len_ - 1) x C_
  EXPECT_EQ(this->output_->shape(0), this->T_);
  EXPECT_EQ(this->output_->shape(1), this->N_);
  EXPECT_EQ(this->output_->shape(2), this->seg_ptr_len_ - 1);
  EXPECT_EQ(this->output_->shape(3), this->C_);
}

TYPED_TEST(RowPoolingLayerTest, TestRowPooling) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  RowPoolingLayer<Dtype>layer(layer_param);
      
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  
  const Dtype* input_array = this->input_->cpu_data();
  const Dtype* output_array = this->output_->cpu_data();
  
  std::stringstream buffer;
  
  buffer << "Input:" << std::endl;
  this->printMat(buffer, input_array, this->C_, this->input_->count());
  buffer << std::endl;
  
  buffer << "Output:" << std::endl;
  this->printMat(buffer, output_array, this->C_, this->output_->count());
  
  LOG(ERROR) << buffer.str();
}


TYPED_TEST(RowPoolingLayerTest, TestRowPoolingGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  RowPoolingLayer<Dtype> layer(layer_param);
      
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradient(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
