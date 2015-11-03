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
class MaskedPoolingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  MaskedPoolingLayerTest() {
    
    blob_image_ = new Blob<Dtype>();
  	blob_seg_inds_ = new Blob<Dtype>();
  	blob_mask_ = new Blob<Dtype>();
  	blob_nseg_ = new Blob<Dtype>();
  	blob_X_ = new Blob<Dtype>();
  
  
    int N = 2;
    int C = 3;
	int max_nseg = 4;
	int mask_lenght = 10;
	
    // fill the values
    Caffe::set_random_seed(1701);
    
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    
    // blob_image shape: N_ x c_ x h_ x w_
    vector<int> image_shape;
    image_shape.push_back(N);
    image_shape.push_back(C);
    image_shape.push_back(5);
    image_shape.push_back(6);
    blob_image_->Reshape(image_shape);
    filler.Fill(blob_image_);
    
    //shape: N_ x max_lenght
    vector<int> mask_shape;
    mask_shape.push_back(N);
    mask_shape.push_back(mask_lenght);
    blob_mask_->Reshape(mask_shape);
    Dtype* mask_data = blob_mask_->mutable_cpu_data();
    //mask for image 1, seg 1
    mask_data[0] = 0;
    mask_data[1] = 1;
    mask_data[2] = 2;
    //mask for image 1, seg 2
    mask_data[3] = 0;
    mask_data[4] = 2;
    mask_data[5] = 7;
    mask_data[6] = 14;
    
    mask_data += mask_lenght;
    //mask for image 2, seg 1
    mask_data[0] = 0;
    mask_data[1] = 2;
    //mask for image 2, seg 2
    mask_data[2] = 7;
    mask_data[3] = 14;
    //mask for image 2, seg 3
    mask_data[4] = 0;
    mask_data[5] = 7;
    
    
    //shape: N_ x max_nseg_
    vector<int> seg_inds_shape;
    seg_inds_shape.push_back(N);
    seg_inds_shape.push_back(max_nseg);
    blob_seg_inds_->Reshape(seg_inds_shape);
    Dtype* seg_inds_data = blob_seg_inds_->mutable_cpu_data();
    //inds for image 1
    seg_inds_data[0] = 0;
    seg_inds_data[1] = 3;
    seg_inds_data[2] = 7;
    
    seg_inds_data += max_nseg;
    
    //inds for image 2
    seg_inds_data[0] = 0;
    seg_inds_data[1] = 2;
    seg_inds_data[2] = 4;
    seg_inds_data[3] = 6;
    
    //shape: N_
    vector<int> nseg_shape;
    nseg_shape.push_back(N);
    blob_nseg_->Reshape(nseg_shape);
    Dtype* nseg_data = blob_nseg_->mutable_cpu_data();
    nseg_data[0] = 2;
    nseg_data[1] = 3;
    
    //Add blobs to input vector
    blob_bottom_vec_.push_back(blob_image_);
    blob_bottom_vec_.push_back(blob_mask_);
    blob_bottom_vec_.push_back(blob_seg_inds_);
    blob_bottom_vec_.push_back(blob_nseg_);
    
    //Add top blob to the output vector
    blob_top_vec_.push_back(blob_X_);
  }
  
  virtual ~MaskedPoolingLayerTest() {
    delete blob_image_;
    delete blob_seg_inds_;
    delete blob_mask_;
    delete blob_nseg_;
    delete blob_X_;
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
  
  Blob<Dtype>* blob_image_;
  Blob<Dtype>* blob_seg_inds_;
  Blob<Dtype>* blob_mask_;
  Blob<Dtype>* blob_nseg_;
  Blob<Dtype>* blob_X_;
  
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MaskedPoolingLayerTest, TestDtypesAndDevices);

TYPED_TEST(MaskedPoolingLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  MaskedPoolingLayer<Dtype>layer(layer_param);
      
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  
  //N_ x max_nseg_ x channels
  EXPECT_EQ(this->blob_X_->shape(0), 2);
  EXPECT_EQ(this->blob_X_->shape(1), 4);
  EXPECT_EQ(this->blob_X_->shape(2), 3);
}

TYPED_TEST(MaskedPoolingLayerTest, TestPooling) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  MaskedPoolingLayer<Dtype>layer(layer_param);
      
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  
  const Dtype* X_data = this->blob_X_->cpu_data();
  const Dtype* image_data = this->blob_image_->cpu_data();
  
  std::stringstream buffer;
  
  buffer << "Input Image:" << std::endl;
  for(int i = 0; i < this->blob_image_->count(0,2); ++i) {
  	this->printMat(buffer, image_data, 6, this->blob_image_->offset(0, 1));
  	image_data += this->blob_image_->offset(0, 1);
  	buffer << std::endl;
  }
  buffer << "Feature Output:" << std::endl;
  this->printMat(buffer, X_data, 3, this->blob_X_->count());
  
  LOG(ERROR) << buffer.str();
}

TYPED_TEST(MaskedPoolingLayerTest, TestInverseGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MaskedPoolingLayer<Dtype> layer(layer_param);
      
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradient(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
