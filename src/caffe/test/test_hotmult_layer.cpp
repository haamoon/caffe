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
  class HotMultLayerTest : public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;
    
  protected:
    HotMultLayerTest() {
      blob_bottom_a_ = NULL;
      blob_bottom_b_ = NULL;
      blob_top_ = NULL;
    }
    
    void clear() {
      blob_bottom_vec_.clear();
      blob_top_vec_.clear();
      
      blob_bottom_shape_a_.clear();
      if(blob_bottom_a_ != NULL) {
        delete blob_bottom_a_;
        blob_bottom_a_ = NULL;
      }
      
      blob_bottom_shape_b_.clear();
      if(blob_bottom_b_ != NULL) {
        delete blob_bottom_b_;
        blob_bottom_b_ = NULL;
      }
      
      if(blob_top_ != NULL) {
        delete blob_top_;
        blob_top_ = NULL;
      }
    }
    
    void setbottom(string mode) {
      HotMultParameter* hotmult_param = layer_param_.mutable_hotmult_param();
      
      hotmult_param->set_mode(mode);
      this->clear();
      
      // fill the values
      Caffe::set_random_seed(1101);
      
      FillerParameter filler_param;
      UniformFiller<Dtype> filler(filler_param);
      
      blob_bottom_shape_a_.push_back(2);
      blob_bottom_shape_a_.push_back(2);
      blob_bottom_shape_a_.push_back(4);

      
      blob_bottom_shape_b_.push_back(4);
      blob_bottom_shape_b_.push_back(2);
      blob_bottom_shape_b_.push_back(3);
      
      blob_bottom_a_ = new Blob<Dtype>(blob_bottom_shape_a_);
      blob_bottom_b_ = new Blob<Dtype>(blob_bottom_shape_b_);
      blob_top_ = new Blob<Dtype>();
      
      
      Dtype* a = this->blob_bottom_a_->mutable_cpu_data();
      for(int i = 0; i < 16; i++) {
        if(mode.compare("ROW") == 0) {
            a[i] = i % 2;
        } else {
          a[i] = i % 3;
        }
      }
      filler.Fill(this->blob_bottom_b_);
      

      blob_bottom_vec_.push_back(blob_bottom_a_);
      blob_bottom_vec_.push_back(blob_bottom_b_);
      blob_top_vec_.push_back(blob_top_);
    }
    
    
    virtual ~HotMultLayerTest() {
      this->clear();
    }
    
    Blob<Dtype>* blob_bottom_a_;
    Blob<Dtype>* blob_bottom_b_;
    Blob<Dtype>* blob_top_;
    
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
    vector<int> blob_bottom_shape_a_;
    vector<int> blob_bottom_shape_b_;
    LayerParameter layer_param_;
  };
  
  TYPED_TEST_CASE(HotMultLayerTest, TestDtypesAndDevices);
    
  TYPED_TEST(HotMultLayerTest, TestSetUpRow) {
    typedef typename TypeParam::Dtype Dtype;
     
    this->setbottom("ROW");    
    HotMultLayer<Dtype> layer(this->layer_param_);
   
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    EXPECT_EQ(this->blob_top_->shape(0), 2);
    EXPECT_EQ(this->blob_top_->shape(1), 2);
    EXPECT_EQ(this->blob_top_->shape(2), 4);
    EXPECT_EQ(this->blob_top_->shape(3), 3);
  }
  
  TYPED_TEST(HotMultLayerTest, TestSetUpCol) {
    typedef typename TypeParam::Dtype Dtype;
    
    this->setbottom("COLUMN");    
    HotMultLayer<Dtype> layer(this->layer_param_);
    
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    EXPECT_EQ(this->blob_top_->shape(0), 2);
    EXPECT_EQ(this->blob_top_->shape(1), 2);
    EXPECT_EQ(this->blob_top_->shape(2), 2);
    EXPECT_EQ(this->blob_top_->shape(3), 4);
  }
  
  TYPED_TEST(HotMultLayerTest, TestHotMultRow) {
    typedef typename TypeParam::Dtype Dtype;
    
    this->setbottom("ROW");  
    HotMultLayer<Dtype> layer(this->layer_param_);
    
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    
    const Dtype* data = this->blob_top_->cpu_data();
    const Dtype* in_data_a = this->blob_bottom_a_->cpu_data();
    const Dtype* in_data_b = this->blob_bottom_b_->cpu_data();
    
    
    std::stringstream buffer;
    buffer << "Row test" << endl << "A:" << endl;
    for (int i = 0; i < this->blob_bottom_a_->count(); ++i) {
      if(i % 4 == 0)
        buffer << ';' << endl;
      buffer << in_data_a[i] << ' ';    
    }
    
    buffer << endl << "B:" << endl;
    buffer.clear();
    for (int i = 0; i < this->blob_bottom_b_->count(); ++i) {
      if(i % 3 == 0)
        buffer << ';' << endl;
      buffer << in_data_b[i] << ' ';    
    }
    
    buffer << endl << "OUT:" << endl;
    buffer.clear();
    for (int i = 0; i < this->blob_top_->count(); ++i) {
      if(i % 3 == 0)
        buffer << ';' << endl;
      buffer << data[i] << ' ';    
    }
    LOG(ERROR) << buffer.str();
  }
  
  TYPED_TEST(HotMultLayerTest, TestHotMultCol) {
    typedef typename TypeParam::Dtype Dtype;
    
    this->setbottom("COLUMN");  
    HotMultLayer<Dtype> layer(this->layer_param_);
    
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    
    const Dtype* data = this->blob_top_->cpu_data();
    const Dtype* in_data_a = this->blob_bottom_a_->cpu_data();
    const Dtype* in_data_b = this->blob_bottom_b_->cpu_data();
    
    std::stringstream buffer;
    buffer << "Column TEST" << endl << "A:" << endl;
    for (int i = 0; i < this->blob_bottom_a_->count(); ++i) {
      if(i % 4 == 0)
        buffer << ';' << endl;
      buffer << in_data_a[i] << ' ';    
    }
    
    buffer << endl << "B:" << endl;
    buffer.clear();
    for (int i = 0; i < this->blob_bottom_b_->count(); ++i) {
      if(i % 3 == 0)
        buffer << ';' << endl;
      buffer << in_data_b[i] << ' ';    
    }
    
    buffer << endl << "OUT:" << endl;
    buffer.clear();
    for (int i = 0; i < this->blob_top_->count(); ++i) {
      if(i % 4 == 0)
        buffer << ';' << endl;
      buffer << data[i] << ' ';    
    }
    LOG(ERROR) << buffer.str();
  }
  
 
  
  // Gradient test
  TYPED_TEST(HotMultLayerTest, TestGradientMultRow) {
    typedef typename TypeParam::Dtype Dtype;
    this->setbottom("ROW");  
    HotMultLayer<Dtype> layer(this->layer_param_);
    
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    vector<bool> back_prob;
    back_prob.push_back(false);
    back_prob.push_back(true);
    
    GradientChecker<Dtype> checker(1e-3, 1e-3);
    checker.CheckGradient(&layer, this->blob_bottom_vec_,
                          this->blob_top_vec_, 1);
  }
  
  
  TYPED_TEST(HotMultLayerTest, TestGradientMultCol) {
    typedef typename TypeParam::Dtype Dtype;
    this->setbottom("COLUMN");  
    HotMultLayer<Dtype> layer(this->layer_param_);
    
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    vector<bool> back_prob;
    back_prob.push_back(false);
    back_prob.push_back(true);
    
    GradientChecker<Dtype> checker(1e-3, 1e-3);
    checker.CheckGradient(&layer, this->blob_bottom_vec_,
                          this->blob_top_vec_, 1);
  }
  
  
}  // namespace caffe
