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
  class SuperpixelPoolingLayerTest : public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;
    
  protected:
    SuperpixelPoolingLayerTest() {
      image_ = new Blob<Dtype>();
      spixel_data_ = new Blob<Dtype>();
      mask_size_= new Blob<Dtype>();
      spixel_ptr_ = new Blob<Dtype>();
      spixel_num_ = new Blob<Dtype>();
      output_ = new Blob<Dtype>();
      
      T = 2;
      N = 2;
      int C = 3;
      int h = 6;
      w = 5;
      spixel_data_len = 10;
      spixel_ptr_len = 10;
      
      // fill the values
      Caffe::set_random_seed(1701);
      
      FillerParameter filler_param;
      UniformFiller<Dtype> filler(filler_param);
      
      // image_ shape: T_ x N_ x c_ x h_ x w_
      vector<int> image_shape;
      image_shape.push_back(T);
      image_shape.push_back(N);
      image_shape.push_back(C);
      image_shape.push_back(h);
      image_shape.push_back(w);
      image_->Reshape(image_shape);
      filler.Fill(image_);
      
      //spixle_data_ shape: T_ x N_ x spixel_data_len x 2
      vector<int> spixel_data_shape;
      spixel_data_shape.push_back(T);
      spixel_data_shape.push_back(N);
      spixel_data_shape.push_back(spixel_data_len);
      spixel_data_shape.push_back(2);
      spixel_data_->Reshape(spixel_data_shape);
      
      Dtype* spixel_data_array = spixel_data_->mutable_cpu_data();
      int i = 0;
      for(int num = 0; num < 2; num++) {
        //data for image 1, superpixel 1
        spixel_data_array[i++] = 0;
        spixel_data_array[i++] = 0;
        //
        spixel_data_array[i++] = 0;
        spixel_data_array[i++] = 1;
      
        //data for image 1, superpixel 2
        spixel_data_array[i++] = 2;
        spixel_data_array[i++] = 0;
        //
        spixel_data_array[i++] = 0;
        spixel_data_array[i++] = 3;
      
        spixel_data_array += spixel_data_len * 2;
        i = 0;
      
        //mask for image 2, super pixel 1
        spixel_data_array[i++] = 4;
        spixel_data_array[i++] = 1;
        //
        spixel_data_array[i++] = 4;
        spixel_data_array[i++] = 2;
        //
        //mask for image 2, super pixel 2
        spixel_data_array[i++] = 1;
        spixel_data_array[i++] = 3;
        //
        spixel_data_array[i++] = 1;
        spixel_data_array[i++] = 4;
        //
        //mask for image 2, super pixel 3
        spixel_data_array[i++] = 0;
        spixel_data_array[i++] = 3;
        //
        spixel_data_array[i++] = 0;
        spixel_data_array[i++] = 4;
        //
        spixel_data_array += spixel_data_len * 2;
        i = 0;
      }
      
      
      //spixel_ptr_ shape: T_ x N_ x spixel_ptr_len
      vector<int> spixel_ptr_shape;
      spixel_ptr_shape.push_back(T);
      spixel_ptr_shape.push_back(N);
      spixel_ptr_shape.push_back(spixel_ptr_len);
      spixel_ptr_->Reshape(spixel_ptr_shape);
      
      Dtype* spixel_ptr_array = spixel_ptr_->mutable_cpu_data();
      
      for(int num = 0; num < 2; num++) {
        //inds for image 1
        spixel_ptr_array[0] = 0;
        spixel_ptr_array[1] = 2;
        spixel_ptr_array[2] = 4;
      
        spixel_ptr_array += spixel_ptr_len;
      
        //inds for image 2
        spixel_ptr_array[0] = 0;
        spixel_ptr_array[1] = 2;
        spixel_ptr_array[2] = 4;
        spixel_ptr_array[3] = 6;
      
        spixel_ptr_array += spixel_ptr_len;
      }
      
      //spixel_num_ shape: T_ x N_
      vector<int> spixel_num_shape;
      spixel_num_shape.push_back(T);
      spixel_num_shape.push_back(N);
      spixel_num_->Reshape(spixel_num_shape);
      Dtype* spixel_num_array = spixel_num_->mutable_cpu_data();
      spixel_num_array[0] = 2;
      spixel_num_array[1] = 3;
      spixel_num_array[2] = 2;
      spixel_num_array[3] = 3;
      
      //mask_size_ shape: T_ x N_ x 2
      vector<int> mask_size_shape;
      mask_size_shape.push_back(T);
      mask_size_shape.push_back(N);
      mask_size_shape.push_back(2);
      mask_size_->Reshape(mask_size_shape);
      Dtype* mask_size_array = mask_size_->mutable_cpu_data();
      mask_size_array[0] = h;
      mask_size_array[1] = w;
      
      mask_size_array[2] = h * 2;
      mask_size_array[3] = w;
      
      mask_size_array[4] = h;
      mask_size_array[5] = w * 2;
      
      mask_size_array[6] = h * 2;
      mask_size_array[7] = w * 2;
      
      //Add blobs to input vector
      blob_bottom_vec_.push_back(image_);
      blob_bottom_vec_.push_back(spixel_data_);
      blob_bottom_vec_.push_back(spixel_ptr_);
      blob_bottom_vec_.push_back(spixel_num_);
      blob_bottom_vec_.push_back(mask_size_);
      //Add top blob to the output vector
      blob_top_vec_.push_back(output_);
    }
    
    virtual ~SuperpixelPoolingLayerTest() {
      delete image_;
      delete spixel_data_;
      delete spixel_ptr_;
      delete spixel_num_;
      delete output_;
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
    
    Blob<Dtype>* image_;
    Blob<Dtype>* spixel_data_;
    Blob<Dtype>* spixel_ptr_;
    Blob<Dtype>* spixel_num_;
    Blob<Dtype>* mask_size_;
    Blob<Dtype>* output_;
    int N;
    int T;
    int spixel_data_len;
    int spixel_ptr_len;
    int w;
    
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
  };
  
  TYPED_TEST_CASE(SuperpixelPoolingLayerTest, TestDtypesAndDevices);
  
  TYPED_TEST(SuperpixelPoolingLayerTest, TestSetUp) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    
    SuperpixelPoolingLayer<Dtype>layer(layer_param);
    
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    
    //T_ x N_ x (spixel_ptr_len - 1) x channels
    EXPECT_EQ(this->output_->shape(0), 2);
    EXPECT_EQ(this->output_->shape(1), 2);
    EXPECT_EQ(this->output_->shape(2), 9);
    EXPECT_EQ(this->output_->shape(3), 3);
  }
  
  TYPED_TEST(SuperpixelPoolingLayerTest, TestSuperpixelPooling) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    
    SuperpixelPoolingLayer<Dtype>layer(layer_param);
    
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    
    const Dtype* spixel_ptr_array = this->spixel_ptr_->cpu_data();
    const Dtype* spixel_data_array = this->spixel_data_->cpu_data();
    const Dtype* spixel_num_array = this->spixel_num_->cpu_data();
    const Dtype* output_array = this->output_->cpu_data();
    const Dtype* image_array = this->image_->cpu_data();
    const Dtype* mask_size_array = this->mask_size_->cpu_data();
    
    
    for(int n = 0; n < this->N * this->T; n++) {
      for(int sp = 0; sp < spixel_num_array[n]; sp++) {
        int start = spixel_ptr_array[n * this->spixel_ptr_len + sp];
        int row1 = spixel_data_array[n * this->spixel_data_len * 2 + start];
        int col1 = spixel_data_array[n * this->spixel_data_len * 2 + start + 1];
        int c_row1 = (int) (row1 * this->image_->shape(4) / mask_size_array[n * 2]);
        int c_col1 = (int) (col1 * this->image_->shape(4) / mask_size_array[n * 2 + 1]);
        LOG(ERROR) << "row1 = " << row1 << ", " << c_row1 << " col1 = " << col1 << ", " << c_col1;
        
        start++;
        
        int row2 = spixel_data_array[n * this->spixel_data_len * 2 + start];
        int col2 = spixel_data_array[n * this->spixel_data_len * 2 + start + 1];
        int c_row2 = (int) (row2 * this->image_->shape(4) / mask_size_array[n * 2]);
        int c_col2 = (int) (col2 * this->image_->shape(4) / mask_size_array[n * 2 + 1]);
        LOG(ERROR) << "row2 = " << row2 << ", " << c_row2 << " col2 = " << col2 << ", " << c_col2;
        
        for(int c = 0; c < 3; c++) {
          Dtype sum = image_array[n * this->image_->count(2) +
                                  c * this->image_->count(3) +
                                  c_row1 * this->w + c_col1] +
                      image_array[n * this->image_->count(2) +
                                  c * this->image_->count(3) +
                                  c_row2 * this->w + c_col2];
          EXPECT_EQ(sum, output_array[n * this->output_->count(2) + sp * 3 + c]);
        }
      }
    }

    std::stringstream buffer;
    
    buffer << "Input Image:" << std::endl;
    this->printMat(buffer, image_array, 5, this->image_->count());
    buffer << "Feature Output:" << std::endl;
    this->printMat(buffer, output_array, 3, this->output_->count());
    
    LOG(ERROR) << buffer.str();
  }
  
  
//  TYPED_TEST(SuperpixelPoolingLayerTest, TestSuperpixelPoolingGradient) {
//    typedef typename TypeParam::Dtype Dtype;
//    LayerParameter layer_param;
//    SuperpixelPoolingLayer<Dtype> layer(layer_param);
//    
//    GradientChecker<Dtype> checker(1e-2, 1e-2);
//    checker.CheckGradient(&layer, this->blob_bottom_vec_,
//                          this->blob_top_vec_, 0);
//  }
  
}  // namespace caffe
