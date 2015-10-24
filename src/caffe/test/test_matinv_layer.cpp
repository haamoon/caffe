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
class MatInvLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  MatInvLayerTest()
      : blob_bottom_a_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {
    
    // fill the values
    Caffe::set_random_seed(1701);
    
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    
    blob_bottom_shape_.push_back(2);
    blob_bottom_shape_.push_back(3);
    blob_bottom_shape_.push_back(3);
    
    blob_bottom_a_->Reshape(blob_bottom_shape_);
    
    filler.Fill(this->blob_bottom_a_);
    blob_bottom_vec_.push_back(blob_bottom_a_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MatInvLayerTest() {
    delete blob_bottom_a_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_a_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<int> blob_bottom_shape_;
};

TYPED_TEST_CASE(MatInvLayerTest, TestDtypesAndDevices);

TYPED_TEST(MatInvLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MatInvParameter* matinv_param = layer_param.mutable_matinv_param();
  matinv_param->set_lambda(.1);
  shared_ptr<MatInvLayer<Dtype> > layer(
      new MatInvLayer<Dtype>(layer_param));
      
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), 2);
  EXPECT_EQ(this->blob_top_->shape(1), 3);
  EXPECT_EQ(this->blob_top_->shape(2), 3);
}

TYPED_TEST(MatInvLayerTest, TestInverse) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MatInvParameter* matinv_param = layer_param.mutable_matinv_param();
  matinv_param->set_lambda(1.0);
  MatInvLayer<Dtype> layer(layer_param);
  
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  
  const Dtype* data = this->blob_top_->cpu_data();
  const Dtype* in_data_a = this->blob_bottom_a_->cpu_data();
  
  for (int n = 0; n < 2; ++n) {
    LOG(INFO) << "Matrix number " << n;
    for (int i = 0; i < 3; ++i) { 
    	LOG(INFO) << '[' << *(in_data_a + n * 9 + i * 3) << ' '
    					 << *(in_data_a + n * 9 + i * 3 + 1)  << ' '
    					 << *(in_data_a + n * 9 + i * 3 + 2) << "] ["
    					 << *(data + n * 9 + i * 3) << ' '
    					 << *(data + n * 9 + i * 3 + 1) << ' '
    					 << *(data + n * 9 + i * 3 + 2) << ']';
    				 
  	}
  }
}

TYPED_TEST(MatInvLayerTest, TestInverseGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MatInvParameter* matinv_param = layer_param.mutable_matinv_param();
  matinv_param->set_lambda(1.0);
  MatInvLayer<Dtype> layer(layer_param);
      
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradient(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
