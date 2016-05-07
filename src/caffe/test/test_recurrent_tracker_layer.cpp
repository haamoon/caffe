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
  class RecurrentTrackerLayerTest : public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;
    
  protected:
    RecurrentTrackerLayerTest() {
      x_ = NULL;
      v_ = NULL;
      cont_ = NULL;
      y_ = NULL;
    }
    
    void clear() {
      if(x_ != NULL) {
        delete x_;
        x_ = NULL;
        x_shape_.clear();
      }
      
      if(v_ != NULL) {
        delete v_;
        v_ = NULL;
        v_shape_.clear();
      }

      if(cont_ != NULL) {
        delete cont_;
        cont_ = NULL;
        cont_shape_.clear();
      }
      
      if(y_ != NULL) {
        delete y_;
        y_ = NULL;
      }
      
      
      blob_bottom_vec_.clear();
      blob_top_vec_.clear();
    }
    
    
    void setbottom(vector<int> x_shape, int num_track = 5, float lambda = .5, float alpha = .5) {
      RecurrentTrackerParameter* recurrent_tracker_param = layer_param_.mutable_recurrent_tracker_param();
      recurrent_tracker_param->set_feature_dim(x_shape[3]);
      recurrent_tracker_param->set_max_ntrack(num_track);
      recurrent_tracker_param->set_lambda(lambda);
      recurrent_tracker_param->set_alpha(alpha);
      this->num_track_ = num_track;
      this->lambda_ = lambda;
      this->alpha_ = alpha;
      this->clear();
      
      
      // fill the values
      Caffe::set_random_seed(1101);
      
      FillerParameter filler_param;
      UniformFiller<Dtype> filler(filler_param);
      
      //x_shape_: T x N x num_seg x num_dim
      x_shape_ = x_shape;

      //v_shape_: T x N x num_seg x num_track
      v_shape_.push_back(x_shape_[0]);
      v_shape_.push_back(x_shape_[1]);
      v_shape_.push_back(x_shape_[2]);
      v_shape_.push_back(num_track);

      //cont_shape_: T x N
      cont_shape_.push_back(x_shape_[0]);
      cont_shape_.push_back(x_shape_[1]);
      
      x_ = new Blob<Dtype>(x_shape);
      v_ = new Blob<Dtype>(v_shape_);
      cont_ = new Blob<Dtype>(cont_shape_);
      
      y_ = new Blob<Dtype>();
      
      filler.Fill(x_);
      filler.Fill(v_);
      
      
      //Dtype* overlap_data = overlaps_->mutable_cpu_data();
      
      //for(int n = 0; n < x_shape_[0] * x_shape_[1]; n++) {
     //   for(int i = 0; i < x_shape_[2]; i++) {
     //     overlap_data[n * x_shape_[2] * x_shape_[2] + i * x_shape_[2] + i] = 1;
     //   }
     // }
      
      Dtype* cont_data = cont_->mutable_cpu_data();
      for(int t = 0; t < x_shape_[0]; ++t) {
        for(int n = 0; n < x_shape_[1]; n++) {
          cont_data[ t * x_shape_[1] + n ] = (t == 0
                                              //|| t == (x_shape_[0] - 1) / 2
                                              ) ? (Dtype).0 : (Dtype)1.0;
        }
      }
      
      blob_bottom_vec_.push_back(x_);
      blob_bottom_vec_.push_back(v_);
      blob_bottom_vec_.push_back(cont_);
      blob_top_vec_.push_back(y_);
    }
    
    
    virtual ~RecurrentTrackerLayerTest() {
      this->clear();
    }
    
    template <typename Dtype>
    void printMat(std::stringstream& buffer, Dtype* mat, int col, int count) {
      buffer << std::fixed << std::setprecision(5);
      for (int i = 0; i < count; ++i) {
        if(i != 0 && i % col == 0) {
          buffer << ';' << endl;
        }
        buffer << std::setw(10) << mat[i] << ' ';
      }
      buffer << endl;
    }
    
    Blob<Dtype>* x_;
    Blob<Dtype>* v_;
    Blob<Dtype>* cont_;
    Blob<Dtype>* y_;
    
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
    
    vector<int> x_shape_;
    vector<int> v_shape_;
    vector<int> cont_shape_;
    
    int num_track_;
    float lambda_;
    float alpha_;
    
    LayerParameter layer_param_;
  };
  
  
  TYPED_TEST_CASE(RecurrentTrackerLayerTest, TestDtypesAndDevices);
  
  TYPED_TEST(RecurrentTrackerLayerTest, TestSetUp) {
    typedef typename TypeParam::Dtype Dtype;
    vector<int> x_shape;
    int T = 10;
    int N = 15;
    int num_seg = 6;
    int num_dim = 3;
    int num_track = 5;
    float lambda = .5;
    float alpha = .5;
    
    x_shape.push_back(T);
    x_shape.push_back(N);
    x_shape.push_back(num_seg);
    x_shape.push_back(num_dim);
    this->setbottom(x_shape, num_track, lambda, alpha);
    
    RecurrentTrackerLayer<Dtype> recurrent_tracker_layer(this->layer_param_);
    
    recurrent_tracker_layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

    //y has: T x N x num_seg x num_track
    EXPECT_EQ(this->y_->shape(0), T);
    EXPECT_EQ(this->y_->shape(1), N);
    EXPECT_EQ(this->y_->shape(2), num_seg);
    EXPECT_EQ(this->y_->shape(3), num_track);
  }
}  // namespace caffe
