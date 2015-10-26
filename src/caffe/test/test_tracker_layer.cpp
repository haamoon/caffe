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
class TrackerLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  TrackerLayerTest() {
  	blob_bottom_a_ = NULL;
  	blob_bottom_b_ = NULL;
  	blob_top_ = NULL;
  }
  
  void clear() {
  	if(x_ != NULL) {
    	delete x_;
    	x_ = NULL;
    	x_shape_.clear();	
    }
    
    if(cont_ != NULL) {
    	delete x_;
    	cont_ = NULL;
    	cont_shape_.clear();	
    }
    
    if( h_0_ != NULL) {
    	delete h_0_;
    	h_0_ = NULL;
    	h_0_shape_.clear();
    }
    
    if(c_0_ != NULL) {
    	delete c_0_;
    	c_0_ = NULL;
    	c_0_shape_.clear();
    }
    
    if(h_T_ != NULL) {
    	delete h_T_;
    	h_T_ = NULL;
    }
    
    if( c_T_ != NULL) {
    	delete  c_T_;
    	 c_T_ = NULL;
    }
    
    if(v_ != NULL) {
    	delete v_;
    	v_ = NULL;
    }
    
    blob_bottom_vec_.clear();
    blob_top_vec_clear();
  }
  
  void setbottom(vector<int> x_shape, int num_track = 5, float lambda = .5) {

	TrackerParameter* tracker_param = layer_param_.mutable_tracker_param();
  	tracker_param->add_lambda(lambda);
	tracker_param->add_num_track(num_track);
	tracker_param->add_feature_dim(x_shape[3]);
	
	this->num_track_ = num_track;
	this->clear();

	// fill the values
    Caffe::set_random_seed(1101);
    
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    
    //x_shape_: T x N x num_seg x num_dim
    x_shape_ = x_shape;
    
    //h_0_shape_: 1 x N x num_dim x num_dim
    h_0_shape_.push_back(1);
    h_0_shape_.push_back(x_shape_[1]);
    h_0_shape_.push_back(x_shape_[3]);
    h_0_shape_.push_back(x_shape_[3]);
    
    //c_0_shape_: 1 x N x num_dim x num_track 
    c_0_shape_.push_back(1);
    c_0_shape_.push_back(x_shape_[1]);
    c_0_shape_.push_back(x_shape_[3]);
    c_0_shape_.push_back(num_track);
    
    x_ = new Blob<Dtype>(x_shape);
    cont_ = new Blob<int>(x_shape);
    h_0_ = new Blob<Dtype>(h_0_shape_);
    c_0_ = new Blob<Dtype>(c_0_shape_);
    h_T_ = new Blob<Dtype>();
    c_T_ = new Blob<Dtype>();
    v_ = new Blob<Dtype>();
    
    filler.Fill(x_);
    filler.Fill(c_0_);
    filler.Fill(h_0_);
    
    blob_bottom_vec_.push_back();
    blob_bottom_vec_.push_back();
    blob_bottom_vec_.push_back();
  }
  
  
  virtual ~TrackerLayerTest() {
  	this->clear();
  }
  
  Blob<Dtype>* x_;
  Blob<Dtype>* cont_;
  Blob<Dtype>* h_0_;
  Blob<Dtype>* c_0_;
  Blob<Dtype>* h_T_;
  Blob<Dtype>* c_T_;
  Blob<Dtype>* v_;
  
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  
  vector<int> x_shape_;
  vector<int> cont_shape_;
  vector<int> h_0_shape_;
  vector<int> c_0_shape_;
  
  int num_track_;
  float lambda_;
  
  LayerParameter layer_param_;
};

TYPED_TEST_CASE(TrackerLayerTest, TestDtypesAndDevices);


TYPED_TEST(TrackerLayerTest, TestSetUpFTF) {
  typedef typename TypeParam::Dtype Dtype;
  
  this->setbottom(false, false, true);    
  TrackerLayer<Dtype> layer(this->layer_param_);

  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), 2);
  EXPECT_EQ(this->blob_top_->shape(1), 3);
  EXPECT_EQ(this->blob_top_->shape(2), 2);
}

TYPED_TEST(TrackerLayerTest, TestSetUpFF) {
  typedef typename TypeParam::Dtype Dtype;
  
  this->setbottom(false, false);    
  TrackerLayer<Dtype> layer(this->layer_param_);

  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), 2);
  EXPECT_EQ(this->blob_top_->shape(1), 3);
  EXPECT_EQ(this->blob_top_->shape(2), 2);
}

TYPED_TEST(TrackerLayerTest, TestSetUpDF) {
  typedef typename TypeParam::Dtype Dtype;
  
  this->setbottom(true, false);    
  TrackerLayer<Dtype> layer(this->layer_param_);

  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), 2);
  EXPECT_EQ(this->blob_top_->shape(1), 4);
  EXPECT_EQ(this->blob_top_->shape(2), 2);
}


TYPED_TEST(TrackerLayerTest, TestSetUpDD) {
  typedef typename TypeParam::Dtype Dtype;
 
  this->setbottom(true, false);  
  TrackerLayer<Dtype> layer(this->layer_param_);
 
    
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), 2);
  EXPECT_EQ(this->blob_top_->shape(1), 4);
}

TYPED_TEST(TrackerLayerTest, TestMultFTF) {
  typedef typename TypeParam::Dtype Dtype;
  this->setbottom(false, false, true);  
  TrackerLayer<Dtype> layer(this->layer_param_);
  
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  
  const Dtype* data = this->blob_top_->cpu_data();
  const Dtype* in_data_a = this->blob_bottom_a_->cpu_data();
  const Dtype* in_data_b = this->blob_bottom_b_->cpu_data();
  
  
  std::stringstream buffer;
  buffer << "FTF TEST" << endl << "A:" << endl;
  for (int i = 0; i < this->blob_bottom_a_->count(); ++i) {
  	if(i % 3 == 0)
  		buffer << ';' << endl;
	buffer << in_data_a[i] << ' ';    
  }
  
  buffer << endl << "B:" << endl;
  buffer.clear();
  for (int i = 0; i < this->blob_bottom_b_->count(); ++i) {
  	if(i % 2 == 0)
  		buffer << ';' << endl;
	buffer << in_data_b[i] << ' ';    
  }
  
  buffer << endl << "OUT:" << endl;
  buffer.clear();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
	if(i % 2 == 0)
  		buffer << ';' << endl;
	buffer << data[i] << ' ';    
  }
  LOG(INFO) << buffer.str();
}

TYPED_TEST(TrackerLayerTest, TestMultFF) {
  typedef typename TypeParam::Dtype Dtype;
  this->setbottom(false, false);  
  TrackerLayer<Dtype> layer(this->layer_param_);
  
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  
  const Dtype* data = this->blob_top_->cpu_data();
  const Dtype* in_data_a = this->blob_bottom_a_->cpu_data();
  const Dtype* in_data_b = this->blob_bottom_b_->cpu_data();
  
  
  std::stringstream buffer;
  buffer << "FF TEST" << endl << "A:" << endl;
  for (int i = 0; i < this->blob_bottom_a_->count(); ++i) {
  	if(i % 4 == 0)
  		buffer << ';' << endl;
	buffer << in_data_a[i] << ' ';    
  }
  
  buffer << endl << "B:" << endl;
  buffer.clear();
  for (int i = 0; i < this->blob_bottom_b_->count(); ++i) {
  	if(i % 2 == 0)
  		buffer << ';' << endl;
	buffer << in_data_b[i] << ' ';    
  }
  
  buffer << endl << "OUT:" << endl;
  buffer.clear();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
	if(i % 2 == 0)
  		buffer << ';' << endl;
	buffer << data[i] << ' ';    
  }
  LOG(INFO) << buffer.str();
}

TYPED_TEST(TrackerLayerTest, TestMultDF) {
  typedef typename TypeParam::Dtype Dtype;
  this->setbottom(true, false);  
  TrackerLayer<Dtype> layer(this->layer_param_);
  
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  
  const Dtype* data = this->blob_top_->cpu_data();
  const Dtype* in_data_a = this->blob_bottom_a_->cpu_data();
  const Dtype* in_data_b = this->blob_bottom_b_->cpu_data();
  
  
  std::stringstream buffer;
  buffer << "DF TEST" << endl << "A:" << endl;
  for (int i = 0; i < this->blob_bottom_a_->count(); ++i) {
  	if(i % 4 == 0)
  		buffer << ';' << endl;
	buffer << in_data_a[i] << ' ';    
  }
  
  buffer << endl << "B:" << endl;
  buffer.clear();
  for (int i = 0; i < this->blob_bottom_b_->count(); ++i) {
  	if(i % 2 == 0)
  		buffer << ';' << endl;
	buffer << in_data_b[i] << ' ';    
  }
  
  buffer << endl << "OUT:" << endl;
  buffer.clear();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
	if(i % 2 == 0)
  		buffer << ';' << endl;
	buffer << data[i] << ' ';    
  }
  LOG(INFO) << buffer.str();
}

TYPED_TEST(TrackerLayerTest, TestMultDD) {
  typedef typename TypeParam::Dtype Dtype;
  this->setbottom(true, true);  
  TrackerLayer<Dtype> layer(this->layer_param_);
  
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  
  const Dtype* data = this->blob_top_->cpu_data();
  const Dtype* in_data_a = this->blob_bottom_a_->cpu_data();
  const Dtype* in_data_b = this->blob_bottom_b_->cpu_data();
  
  
  std::stringstream buffer;
  buffer << "DD TEST" << endl << "A:" << endl;
  for (int i = 0; i < this->blob_bottom_a_->count(); ++i) {
  	if(i % 4 == 0)
  		buffer << ';' << endl;
	buffer << in_data_a[i] << ' ';    
  }
  
  buffer << endl << "B:" << endl;
  buffer.clear();
  for (int i = 0; i < this->blob_bottom_b_->count(); ++i) {
  	if(i % 4 == 0)
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
  LOG(INFO) << buffer.str();
}

//Gradient test
TYPED_TEST(TrackerLayerTest, TestGradientFTF) {
  typedef typename TypeParam::Dtype Dtype;
  this->setbottom(false, false, true);  
  TrackerLayer<Dtype> layer(this->layer_param_);
  
 GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradient(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}


TYPED_TEST(TrackerLayerTest, TestGradientFF) {
  typedef typename TypeParam::Dtype Dtype;
  this->setbottom(false, false);  
  TrackerLayer<Dtype> layer(this->layer_param_);
  
 GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradient(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(TrackerLayerTest, TestGradientDF) {
  typedef typename TypeParam::Dtype Dtype;
  this->setbottom(true, false);  
  TrackerLayer<Dtype> layer(this->layer_param_);
  
 GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradient(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(TrackerLayerTest, TestGradientDD) {
  typedef typename TypeParam::Dtype Dtype;
  this->setbottom(true, true);  
  TrackerLayer<Dtype> layer(this->layer_param_);
  
 GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradient(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}
}  // namespace caffe
