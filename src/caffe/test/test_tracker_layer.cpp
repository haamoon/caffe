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
      overlaps_ = NULL;
      seg_num_ = NULL;
      cont_ = NULL;
      v_ = NULL;
      vtilde_ = NULL;
    }
    
    void clear() {
      if(x_ != NULL) {
        delete x_;
        x_ = NULL;
        x_shape_.clear();
      }
      
      if(overlaps_ != NULL) {
        delete overlaps_;
        overlaps_ = NULL;
        overlaps_shape_.clear();
      }
      
      if(seg_num_ != NULL) {
        delete seg_num_;
        seg_num_ = NULL;
        seg_num_shape_.clear();
      }
      
      if(cont_ != NULL) {
        delete x_;
        cont_ = NULL;
        cont_shape_.clear();
      }
      
      if(v_ != NULL) {
        delete v_;
        v_ = NULL;
      }
      
      if(vtilde_ != NULL) {
        delete vtilde_;
        vtilde_ = NULL;
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
      
      //seg_num_shape_: T x N
      seg_num_shape_.push_back(x_shape_[0]);
      seg_num_shape_.push_back(x_shape_[1]);
      
      //overlaps_shape_: T X N X num_seg X num_seg
      overlaps_shape_.push_back(x_shape_[0]);
      overlaps_shape_.push_back(x_shape_[1]);
      overlaps_shape_.push_back(x_shape_[2]);
      overlaps_shape_.push_back(x_shape_[2]);
      
      
      x_ = new Blob<Dtype>(x_shape);
      overlaps_ = new Blob<Dtype>(overlaps_shape_);
      seg_num_ = new Blob<Dtype>(seg_num_shape_);
      cont_ = new Blob<Dtype>(cont_shape_);
      
      v_ = new Blob<Dtype>();
      vtilde_ = new Blob<Dtype>();
      
      filler.Fill(x_);
      
      Dtype* overlap_data = overlaps_->mutable_cpu_data();
      
      for(int i = 0; i < x_shape_[2] * x_shape_[0] * x_shape_[1]; i++) {
        for(int j = 0; j < x_shape_[2]; j++) {
          overlap_data[i * x_shape_[2] + j] = i;
        }
      }
      
      
      Dtype* cont_data = cont_->mutable_cpu_data();
      Dtype* seg_num_data = seg_num_->mutable_cpu_data();
      for(int t = 0; t < x_shape_[0]; ++t) {
        for(int n = 0; n < x_shape_[1]; n++) {
          cont_data[ t * x_shape_[1] + n ] = (t == 0
                                              //|| t == (x_shape_[0] - 1) / 2
                                              ) ? (Dtype).0 : (Dtype)1.0;
          seg_num_data[ t * x_shape_[1] + n ] = x_shape_[2];
        }
      }
      
      blob_bottom_vec_.push_back(x_);
      blob_bottom_vec_.push_back(overlaps_);
      blob_bottom_vec_.push_back(seg_num_);
      blob_bottom_vec_.push_back(cont_);
      blob_top_vec_.push_back(v_);
      blob_top_vec_.push_back(vtilde_);
    }
    
    
    virtual ~TrackerLayerTest() {
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
    Blob<Dtype>* overlaps_;
    Blob<Dtype>* seg_num_;
    Blob<Dtype>* cont_;
    Blob<Dtype>* v_;
    Blob<Dtype>* vtilde_;
    
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
    
    vector<int> x_shape_;
    vector<int> overlaps_shape_;
    vector<int> seg_num_shape_;
    vector<int> cont_shape_;
    
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
    
    //v has: T x N x num_track x num_seg
    EXPECT_EQ(this->v_->shape(0), T);
    EXPECT_EQ(this->v_->shape(1), N);
    EXPECT_EQ(this->v_->shape(2), num_track);
    EXPECT_EQ(this->v_->shape(3), num_seg);
    
    
    //v has: T x N x num_track x num_seg
    EXPECT_EQ(this->vtilde_->shape(0), T);
    EXPECT_EQ(this->vtilde_->shape(1), N);
    EXPECT_EQ(this->vtilde_->shape(2), num_track);
    EXPECT_EQ(this->vtilde_->shape(3), num_seg);
  }
  
  TYPED_TEST(TrackerLayerTest, TestForwardT1) {
    typedef typename TypeParam::Dtype Dtype;
    vector<int> x_shape;
    
    int T = 2;
    int N = 1;
    int num_seg = 10;
    int num_dim = 3;
    int num_track = 4;
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
    buffer << "X = [ " << endl;
    this->printMat(buffer, this->x_->cpu_data(), num_dim, this->x_->count());
    buffer << "];" << endl << "overlaps = [" << endl;
    this->printMat(buffer, this->overlaps_->cpu_data(), num_seg, this->overlaps_->count());
    buffer << "];" << endl << "seg_nums = [" << endl;
    this->printMat(buffer, this->seg_num_->cpu_data(), 1, this->seg_num_->count());
    
    buffer << "];" << endl << "cont = [" << endl;
    this->printMat(buffer, this->cont_->cpu_data(), N, this->cont_->count());
  
    buffer << "];" << endl << "Vr = [" << endl;
    this->printMat(buffer, this->v_->cpu_data(), num_seg, this->v_->count());
    
    buffer << "];" << endl << "Vtilder = [" << endl;
    this->printMat(buffer, this->vtilde_->cpu_data(), num_seg, this->vtilde_->count());
    buffer << "];" << endl;
    buffer << "T = " << T << ";" << endl;
    buffer << "N = " << N << ";" << endl;
    buffer << "lambda = " << lambda << ";" << endl;
    buffer << "num_track = " << num_track << ";" << endl;
    
    LOG(ERROR) << buffer.str();
  }
  
}  // namespace caffe
