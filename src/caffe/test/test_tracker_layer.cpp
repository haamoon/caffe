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
  	x_ = NULL;
  	cont_ = NULL;
  	h_0_ = NULL;
  	c_0_ = NULL;
  	h_T_ = NULL;
  	c_T_ = NULL;
  	v_ = NULL;
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
    blob_top_vec_.clear();
  }
  
  
  void setbottom(vector<int> x_shape, int num_track = 5, float lambda = .5) {
	
	TrackerParameter* tracker_param = layer_param_.mutable_tracker_param();
  	tracker_param->set_lambda(lambda);
	tracker_param->set_num_track(num_track);
	tracker_param->set_feature_dim(x_shape[3]);
	
	this->num_track_ = num_track;
	this->clear();

	// fill the values
    Caffe::set_random_seed(1101);
    
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    
    //x_shape_: T x N x num_seg x num_dim
    x_shape_ = x_shape;
    
    //cont_shape_: T x N 
    cont_shape_.push_back(x_shape_[0]);
    cont_shape_.push_back(x_shape_[1]);
    
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
    cont_ = new Blob<Dtype>(cont_shape_);
    //h_0_ = new Blob<Dtype>(h_0_shape_);
    //c_0_ = new Blob<Dtype>(c_0_shape_);
    //h_T_ = new Blob<Dtype>();
    //c_T_ = new Blob<Dtype>();
    v_ = new Blob<Dtype>();
    
    filler.Fill(x_);
    //filler.Fill(c_0_);
    //filler.Fill(h_0_);
    
    Dtype* cont_data = cont_->mutable_cpu_data();
    for(int t = 0; t < x_shape_[0]; ++t) {
    	for(int n = 0; n < x_shape_[1]; n++) {
    		cont_data[ t * x_shape_[1] + n ] = (t == 0 
    		//|| t == (x_shape_[0] - 1) / 2
    		) ? (Dtype).0 : (Dtype)1.0;
    	}
    }
    
    blob_bottom_vec_.push_back(x_);
    blob_bottom_vec_.push_back(cont_);
    //blob_bottom_vec_.push_back(c_0_);
	//blob_bottom_vec_.push_back(h_0_);
	
	blob_top_vec_.push_back(v_);
  }
  
  
  virtual ~TrackerLayerTest() {
  	this->clear();
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


TYPED_TEST(TrackerLayerTest, TestSetUp) {
 	typedef typename TypeParam::Dtype Dtype;
	vector<int> x_shape;
	
	int T = 10;
	int N = 15;
	int num_seg = 4;
	int num_dim = 3;
	int num_track = 5;
	float lambda = .5;
	
	x_shape.push_back(T);
	x_shape.push_back(N);
	x_shape.push_back(num_seg);
	x_shape.push_back(num_dim);
	this->setbottom(x_shape, num_track, lambda);
	
	TrackerLayer<Dtype> tracker_layer(this->layer_param_);
	
	tracker_layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  	
  	//v has: T x N x num_seg x num_track
  	EXPECT_EQ(this->v_->shape(0), T);
  	EXPECT_EQ(this->v_->shape(1), N);
  	EXPECT_EQ(this->v_->shape(2), num_seg);
  	EXPECT_EQ(this->v_->shape(3), num_track);
}

TYPED_TEST(TrackerLayerTest, TestForwardT1) {
 	typedef typename TypeParam::Dtype Dtype;
	vector<int> x_shape;
	
	int T = 3;
	int N = 2;
	int num_seg = 4;
	int num_dim = 3;
	int num_track = 5;
	float lambda = .5;
	
	x_shape.push_back(T);
	x_shape.push_back(N);
	x_shape.push_back(num_seg);
	x_shape.push_back(num_dim);
	this->setbottom(x_shape, num_track, lambda);
	
	TrackerLayer<Dtype> tracker_layer(this->layer_param_);
	
	tracker_layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  	
  	tracker_layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  
  	std::stringstream buffer;	
  	buffer << "x: " << endl;
  	this->printMat(buffer, this->x_->cpu_data(), num_dim, this->x_->count());
  	buffer << "cont: " << endl;
  	this->printMat(buffer, this->cont_->cpu_data(), N, this->cont_->count());
  	buffer << "v: " << endl;
  	this->printMat(buffer, this->v_->cpu_data(), num_track, this->v_->count());
	LOG(ERROR) << buffer.str();
}

}  // namespace caffe
