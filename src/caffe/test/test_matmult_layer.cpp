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
class MatMultLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  MatMultLayerTest() {
  	blob_bottom_a_ = NULL;
  	blob_bottom_b_ = NULL;
  	blob_top_ = NULL;
  }
  
  void clear() {
  	if(blob_bottom_a_ != NULL) {
    	delete blob_bottom_a_;
    	blob_bottom_a_ = NULL;
    }
    
    if(blob_bottom_b_ != NULL) {
    	delete blob_bottom_b_;
    	blob_bottom_b_ = NULL;
    }
    
    if(blob_top_ != NULL) {
    	delete blob_top_;
    	blob_top_ = NULL;
    }
  }
  
  void setbottom(bool diag_a, bool diag_b, bool trans_a = false, bool trans_b = false) {

	MatMultParameter* matmult_param = layer_param_.mutable_matmult_param();
  	matmult_param->add_diagonal_input(diag_a);
  	matmult_param->add_diagonal_input(diag_b);


	matmult_param->set_transpose_a(trans_a);
        matmult_param->set_transpose_b(trans_b);
	this->clear();

	// fill the values
    Caffe::set_random_seed(1101);
    
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    
    if(diag_a) {
    	blob_bottom_shape_a_.push_back(2);
    	blob_bottom_shape_a_.push_back(4);
    } else {
    	if(trans_a == false) {
    		blob_bottom_shape_a_.push_back(2);
    		blob_bottom_shape_a_.push_back(3);
    		blob_bottom_shape_a_.push_back(4);
    	} else {
    		blob_bottom_shape_a_.push_back(2);
    		blob_bottom_shape_a_.push_back(4);
    		blob_bottom_shape_a_.push_back(3);
    	}
    }
    
    if(diag_b) {
    	blob_bottom_shape_b_.push_back(2);
    	blob_bottom_shape_b_.push_back(4);
    } else {
      if(trans_b == false) {
        blob_bottom_shape_b_.push_back(2);
        blob_bottom_shape_b_.push_back(4);
        blob_bottom_shape_b_.push_back(2);
      } else {
        blob_bottom_shape_b_.push_back(2);
        blob_bottom_shape_b_.push_back(2);
        blob_bottom_shape_b_.push_back(4);
      }
    }
    
    blob_bottom_a_ = new Blob<Dtype>(blob_bottom_shape_a_);
    blob_bottom_b_ = new Blob<Dtype>(blob_bottom_shape_b_);
    blob_top_ = new Blob<Dtype>();
    
    filler.Fill(this->blob_bottom_a_);
    filler.Fill(this->blob_bottom_b_);

    blob_bottom_vec_.clear();
    blob_top_vec_.clear();
    blob_bottom_vec_.push_back(blob_bottom_a_);
    blob_bottom_vec_.push_back(blob_bottom_b_);
    blob_top_vec_.push_back(blob_top_);
  }
  
  
  virtual ~MatMultLayerTest() {
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

TYPED_TEST_CASE(MatMultLayerTest, TestDtypesAndDevices);



// TYPED_TEST(MatMultLayerTest, TestSetUpFTF) {
//   typedef typename TypeParam::Dtype Dtype;
//   
//   this->setbottom(false, false, true);    
//   MatMultLayer<Dtype> layer(this->layer_param_);
// 
//   layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//   EXPECT_EQ(this->blob_top_->shape(0), 2);
//   EXPECT_EQ(this->blob_top_->shape(1), 3);
//   EXPECT_EQ(this->blob_top_->shape(2), 2);
// }
// 
// TYPED_TEST(MatMultLayerTest, TestSetUpFF) {
//   typedef typename TypeParam::Dtype Dtype;
//   
//   this->setbottom(false, false);    
//   MatMultLayer<Dtype> layer(this->layer_param_);
// 
//   layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//   EXPECT_EQ(this->blob_top_->shape(0), 2);
//   EXPECT_EQ(this->blob_top_->shape(1), 3);
//   EXPECT_EQ(this->blob_top_->shape(2), 2);
// }
// 
// TYPED_TEST(MatMultLayerTest, TestSetUpDF) {
//   typedef typename TypeParam::Dtype Dtype;
//   
//   this->setbottom(true, false);    
//   MatMultLayer<Dtype> layer(this->layer_param_);
// 
//   layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//   EXPECT_EQ(this->blob_top_->shape(0), 2);
//   EXPECT_EQ(this->blob_top_->shape(1), 4);
//   EXPECT_EQ(this->blob_top_->shape(2), 2);
// }
// 
// 
// TYPED_TEST(MatMultLayerTest, TestSetUpDD) {
//   typedef typename TypeParam::Dtype Dtype;
//  
//   this->setbottom(true, false);  
//   MatMultLayer<Dtype> layer(this->layer_param_);
//  
//     
//   layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//   EXPECT_EQ(this->blob_top_->shape(0), 2);
//   EXPECT_EQ(this->blob_top_->shape(1), 4);
// }

TYPED_TEST(MatMultLayerTest, TestMultFFT) {
  typedef typename TypeParam::Dtype Dtype;
  this->setbottom(false, false, false, true);  
  MatMultLayer<Dtype> layer(this->layer_param_);
  
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  
  const Dtype* data = this->blob_top_->cpu_data();
  const Dtype* in_data_a = this->blob_bottom_a_->cpu_data();
  const Dtype* in_data_b = this->blob_bottom_b_->cpu_data();
  
  
  std::stringstream buffer;
  buffer << "FFT TEST" << endl << "A:" << endl;
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
    if(i % 2 == 0)
      buffer << ';' << endl;
    buffer << data[i] << ' ';    
  }
  LOG(ERROR) << buffer.str();
}

TYPED_TEST(MatMultLayerTest, TestMultFTF) {
  typedef typename TypeParam::Dtype Dtype;
  this->setbottom(false, false, true);  
  MatMultLayer<Dtype> layer(this->layer_param_);
  
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
  LOG(ERROR) << buffer.str();
}

TYPED_TEST(MatMultLayerTest, TestMultFF) {
  typedef typename TypeParam::Dtype Dtype;
  this->setbottom(false, false);  
  MatMultLayer<Dtype> layer(this->layer_param_);
  
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


TYPED_TEST(MatMultLayerTest, TestMultDF) {
  typedef typename TypeParam::Dtype Dtype;
  this->setbottom(true, false);  
  MatMultLayer<Dtype> layer(this->layer_param_);
  
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
  LOG(ERROR) << buffer.str();
}


TYPED_TEST(MatMultLayerTest, TestMultDD) {
  typedef typename TypeParam::Dtype Dtype;
  this->setbottom(true, true);  
  MatMultLayer<Dtype> layer(this->layer_param_);
  
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
  LOG(ERROR) << buffer.str();
}

// Gradient test
TYPED_TEST(MatMultLayerTest, TestGradientFFT) {
  typedef typename TypeParam::Dtype Dtype;
  this->setbottom(false, false, false, true);  
  MatMultLayer<Dtype> layer(this->layer_param_);
  
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradient(&layer, this->blob_bottom_vec_,
                        this->blob_top_vec_);
}


TYPED_TEST(MatMultLayerTest, TestGradientFTF) {
  typedef typename TypeParam::Dtype Dtype;
  this->setbottom(false, false, true);  
  MatMultLayer<Dtype> layer(this->layer_param_);
  
 GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradient(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}


TYPED_TEST(MatMultLayerTest, TestGradientFF) {
  typedef typename TypeParam::Dtype Dtype;
  this->setbottom(false, false);  
  MatMultLayer<Dtype> layer(this->layer_param_);
  
 GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradient(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}


TYPED_TEST(MatMultLayerTest, TestGradientDF) {
  typedef typename TypeParam::Dtype Dtype;
  this->setbottom(true, false);  
  MatMultLayer<Dtype> layer(this->layer_param_);
  
 GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradient(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(MatMultLayerTest, TestGradientDD) {
  typedef typename TypeParam::Dtype Dtype;
  this->setbottom(true, true);  
  MatMultLayer<Dtype> layer(this->layer_param_);
  
 GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradient(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}
}  // namespace caffe
