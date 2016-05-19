#include <vector>

#include "caffe/blob.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/tracker_math.hpp"
#include "caffe/tracking_layers.hpp"


namespace caffe {

template <typename Dtype>
void LinearTrackerLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  vector<int> feature_shape = bottom[0]->shape();
  CHECK_GE(feature_shape.size(), 2);
  input_start_axis_ = feature_shape.size() - 2;
  
  this->layer_param_.mutable_inner_product_param()->set_axis(input_start_axis_ + 1);
  
  max_ntrack_ = this->layer_param_.tracker_param().num_track();
  sample_inter_ = this->layer_param_.tracker_param().sample_inter();
  this->layer_param_.mutable_inner_product_param()->set_num_output(max_ntrack_);
  lambda_ = this->layer_param_.tracker_param().lambda();
  w_scale_ = this->layer_param_.tracker_param().softmax_scale();
  // Intialize the weight
  //vector<int> weight_shape(2);
  //weight_shape[0] = N_;
  //weight_shape[1] = K_;
  //vector<int> bias_shape(1, N_);
  
  //this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
  //this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
  
  InnerProductLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void LinearTrackerLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //bottom[0]: X is a ... x max_nseg_ x dim feature matrix
  //bottom[1]: nseg is a ... vector
  //bottom[2]: cont is a ... vector
  //bottom[3]: ntrack is a ... vector
  //bottom[4]: V is a ... x max_nseg_ x max_ntrack matrix
  
  vector<int> feature_shape = bottom[0]->shape();
  vector<int> v_shape = bottom[4]->shape();
  
  CHECK_GE(feature_shape.size(), 2);
  CHECK_GE(v_shape.size(), 2);
  CHECK_EQ(feature_shape.size() - 2, input_start_axis_);
  
  CHECK_EQ(bottom[0]->count(0, input_start_axis_), 1);
  CHECK_EQ(bottom[1]->count(), 1);
  CHECK_EQ(bottom[2]->count(), 1);
  CHECK_EQ(bottom[3]->count(), 1);
  CHECK_EQ(bottom[4]->count(0, input_start_axis_), 1);
  
  max_nseg_ = feature_shape[input_start_axis_];
  feature_dim_ = feature_shape[input_start_axis_ + 1];
  
  //CHECK_EQ(max_nseg_, v_shape[input_start_axis_]);
  //CHECK_EQ(max_ntrack_, v_shape[input_start_axis_ + 1]);
  
  
  int max_nsample = max_nseg_ / sample_inter_;
  
  vector<int> b_ntxns_shape;
  b_ntxns_shape.push_back(max_ntrack_);
  b_ntxns_shape.push_back(max_nsample);
  //b1_ntrack_nseg_.Reshape(b_ntxns_shape);
  b2_ntrack_nseg_.Reshape(b_ntxns_shape);
  
  vector<int> b_nsxns_shape(2, max_nsample);
  b_nseg_nseg_.Reshape(b_nsxns_shape);
  
  vector<int> o_shape(1, max_nsample);
  o_.Reshape(o_shape);
  
  FillerParameter filler_param;
  filler_param.set_value(1.0);
  filler_param.set_type("constant");
  ConstantFiller<Dtype> constant_filler(filler_param);
  constant_filler.Fill(&o_);
  
  InnerProductLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
void LinearTrackerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int cont = static_cast<int>(*bottom[2]->cpu_data());
  int nseg = static_cast<int>(*bottom[1]->cpu_data());
  int ntrack = static_cast<int>(*bottom[3]->cpu_data());
  
  if(cont == 0 && nseg != 0 && ntrack != 0) {
    const Dtype* X = bottom[0]->cpu_data();
    const Dtype* V = bottom[4]->cpu_data();
    
    Dtype* weight_data = this->blobs_[0]->mutable_cpu_data();
    Dtype* bias_data = this->blobs_[1]->mutable_cpu_data();
    //Dtype* b1_ntxns = this->b1_ntrack_nseg_.mutable_cpu_data();
    Dtype* b2_ntxns = this->b2_ntrack_nseg_.mutable_cpu_data();
    Dtype* b_nsxns = this->b_nseg_nseg_.mutable_cpu_data();
    const Dtype* o_ns = this->o_.cpu_data();
    
    caffe_set(this->blobs_[0]->count(), (Dtype)0.0, weight_data);
    caffe_set(this->N_, (Dtype)0.0, bias_data);
    
    // Compute followings using nsample samples of X and V:
    // weight_data = V' inv(XX' + \lambda I + o o') X
    // bias_data = V' inv(XX' + \lambda I + o o') o
    
    // 1) b_nsxns = inv(XX' + \lambda I + o o')
    int nsample = nseg / sample_inter_;
    
    // b_nsxns = o o'
    caffe_set(nsample * nsample, (Dtype) 1.0, b_nsxns);
    // b_nsxns += \lambda I
    tracker_strided_add_scalar<Dtype>(nsample * nsample, lambda_, nsample + 1, b_nsxns);
    
    
    // b_nsxns += X' X
    tracker_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, nsample, 
                            nsample, feature_dim_, (Dtype) 1.0, X, 
                            feature_dim_ * sample_inter_, X, 
                            feature_dim_ * sample_inter_, (Dtype) 1.0, 
                            b_nsxns);
    
    // b_nsxns = inv(b_nsxns)
    tracker_cpu_inverse<Dtype>(nsample, b_nsxns);
    
    
    // 2) b2_ntxns = V' * bnsxns
    //tracker_cpu_copy<Dtype>(V, sample_inter_, max_nseg_, b1_ntxns, 1, max_nseg_, max_ntrack_, nsample);
    
//     caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, ntrack, 
//                           nsample, nsample, (Dtype) 1.0, b1_ntxns, 
//                           b_nsxns, (Dtype) 0.0, b2_ntxns);
    
    tracker_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, ntrack, 
                            nsample, nsample, (Dtype) 1.0, V, 
                            max_ntrack_ * sample_inter_, b_nsxns, 
                            nsample, (Dtype) 0.0, 
                            b2_ntxns);
    
    // 3) weight_data = b2_ntxns * X
    tracker_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, ntrack, 
                            feature_dim_, nsample, (Dtype) 1.0, b2_ntxns, 
                            nsample, X, 
                            feature_dim_ * sample_inter_, (Dtype) 0.0, 
                            weight_data);
    
    // 3.5) weight_data = scale_w_ * weight_data
    if(w_scale_ != 1.0) {
      caffe_cpu_scale<Dtype>(ntrack * feature_dim_, (Dtype) w_scale_, weight_data, weight_data);
    }
    
    // 4) bias_data = b2_ntxns * o
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, ntrack, 
                          1, nsample, (Dtype) 1.0, b2_ntxns, 
                          o_ns, (Dtype) 0.0, bias_data);
                          
   //std::stringstream buffer;
   //tracker_printMat<Dtype>(buffer, bias_data, ntrack, ntrack);
   //LOG(ERROR) << std::endl << buffer.str();
  }
  InnerProductLayer<Dtype>::Forward_cpu(bottom, top);
}

template <typename Dtype>
void LinearTrackerLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  InnerProductLayer<Dtype>::Backward_cpu(top, propagate_down, bottom);
  
  //add \lambda W to the gradient
  if(lambda_ != 0) {
    caffe_axpy<Dtype>(this->blobs_[0]->count(), lambda_, this->blobs_[0]->cpu_data(), this->blobs_[0]->mutable_cpu_diff());
    caffe_axpy<Dtype>(this->blobs_[1]->count(), lambda_, this->blobs_[1]->cpu_data(), this->blobs_[1]->mutable_cpu_diff());
  }
  
  //int cont = static_cast<int>(*bottom[2]->cpu_data());
  //if(cont == 0) {
  //  caffe_set(bottom[0]->count(), (Dtype) 0.0, bottom[0]->mutable_cpu_diff());
  //}
  
}

#ifdef CPU_ONLY
STUB_GPU(LinearTrackerLayer);
#endif

INSTANTIATE_CLASS(LinearTrackerLayer);
REGISTER_LAYER_CLASS(LinearTracker);

}  // namespace caffe
