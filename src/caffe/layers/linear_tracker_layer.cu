#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/tracker_math.hpp"
#include "caffe/tracking_layers.hpp"
#include "caffe/filler.hpp"


namespace caffe {


template <typename Dtype>
void LinearTrackerLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int cont = static_cast<int>(*bottom[2]->cpu_data());
  int nseg = static_cast<int>(*bottom[1]->cpu_data());
  int ntrack = static_cast<int>(*bottom[3]->cpu_data());
  if(cont == 0 && nseg != 0 && ntrack != 0) {    
    const Dtype* X = bottom[0]->gpu_data();
    const Dtype* V = bottom[4]->gpu_data();
    
    Dtype* weight_data = this->blobs_[0]->mutable_gpu_data();
    Dtype* bias_data = this->blobs_[1]->mutable_gpu_data();
    //Dtype* b1_ntxns = this->b1_ntrack_nseg_.mutable_gpu_data();
    Dtype* b2_ntxns = this->b2_ntrack_nseg_.mutable_gpu_data();
    Dtype* b_nsxns = this->b_nseg_nseg_.mutable_gpu_data();
    const Dtype* o_ns = this->o_.gpu_data();
    
    caffe_gpu_set(this->blobs_[0]->count(), (Dtype)0.0, weight_data);
    caffe_gpu_set(this->N_, (Dtype)0.0, bias_data);
    
    // Compute followings using nsample samples of X and V:
    // weight_data = V' inv(XX' + \lambda I + o o') X
    // bias_data = V' inv(XX' + \lambda I + o o') o
    
    
    int nsample = nseg / sample_inter_;
    
    
    // 1) b_nsxns = inv(XX' + \lambda I + o o')
    // b_nsxns = o o'
    caffe_gpu_set(nsample * nsample, (Dtype) 1.0, b_nsxns);
    // b_nsxns += \lambda I
    tracker_gpu_strided_add_scalar<Dtype>(nsample * nsample, lambda_, nsample + 1, b_nsxns);    
    // b_nsxns += XX'
    tracker_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, nsample, 
                            nsample, feature_dim_, (Dtype) 1.0, X, 
                            feature_dim_ * sample_inter_, X, 
                            feature_dim_ * sample_inter_, (Dtype) 1.0, 
                            b_nsxns);
    // b_nsxns = inv(b_nsxns)
    tracker_gpu_inverse<Dtype>(nsample, b_nsxns);
    
    
    //sometimes inverse returns nan
    Dtype inv_asum = 0;
    caffe_gpu_asum<Dtype>(nsample*nsample, b_nsxns, &inv_asum);
      
    if(!std::isfinite(inv_asum)) {
      LOG(ERROR) << "Got nan in inv_asum..";
      caffe_gpu_set(nsample * nsample, (Dtype) 0.0, b_nsxns);
    }
    
    // 2) b2_ntxns = V' * bnsxns    
    tracker_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, ntrack, 
                            nsample, nsample, (Dtype) 1.0, V, 
                            max_ntrack_ * sample_inter_, b_nsxns, 
                            nsample, (Dtype) 0.0, 
                            b2_ntxns);
    
    // 3) weight_data = b2_ntxns * X
    tracker_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, ntrack, 
                            feature_dim_, nsample, (Dtype) 1.0, b2_ntxns, 
                            nsample, X, 
                            feature_dim_ * sample_inter_, (Dtype) 0.0, 
                            weight_data);
    
    // 3.5) weight_data = scale_w_ * weight_data
    if(w_scale_ != 1.0) {
      caffe_gpu_scale<Dtype>(ntrack * feature_dim_, (Dtype) w_scale_, weight_data, weight_data);
    }
    
    // 4) bias_data = b2_ntxns * o
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, ntrack, 
                          1, nsample, (Dtype) 1.0, b2_ntxns, 
                          o_ns, (Dtype) 0.0, bias_data);
  }
  
  InnerProductLayer<Dtype>::Forward_gpu(bottom, top);
  
}

template <typename Dtype>
void LinearTrackerLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  InnerProductLayer<Dtype>::Backward_gpu(top, propagate_down, bottom);
  
  //add \lambda W to the gradient
  if(lambda_ != 0) {
    caffe_gpu_axpy<Dtype>(this->blobs_[0]->count(), lambda_, this->blobs_[0]->gpu_data(), this->blobs_[0]->mutable_gpu_diff());
    caffe_gpu_axpy<Dtype>(this->blobs_[1]->count(), lambda_, this->blobs_[1]->gpu_data(), this->blobs_[1]->mutable_gpu_diff());
  }
}

template <typename Dtype>
void LinearTrackerLayer<Dtype>::save_and_exit(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {    
    int nseg = static_cast<int>(*bottom[1]->cpu_data());
    
    const Dtype* X = bottom[0]->gpu_data();
    
    //inv(XX' + \lambda I + o o')
    
    int nsample = nseg / sample_inter_;
    
    // 1) b_nsxns = inv(XX' + \lambda I + o o')
    // b_nsxns = o o'
    caffe_gpu_set(nsample * nsample, (Dtype) 1.0, this->b_nseg_nseg_.mutable_gpu_data());
    // b_nsxns += \lambda I
    tracker_gpu_strided_add_scalar<Dtype>(nsample * nsample, lambda_, nsample + 1, this->b_nseg_nseg_.mutable_gpu_data());
      
    tracker_saveMat<Dtype>("matrix_xx_I.bin", this->b_nseg_nseg_.cpu_data(), nsample, nsample * nsample);
      
    // b_nsxns += XX'
    tracker_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, nsample, 
                              nsample, feature_dim_, (Dtype) 1.0, X, 
                              feature_dim_ * sample_inter_, X, 
                              feature_dim_ * sample_inter_, (Dtype) 1.0, 
                              this->b_nseg_nseg_.mutable_gpu_data());
      
    tracker_saveMat<Dtype>("matrix_xx_I.bin", this->b_nseg_nseg_.cpu_data(), nsample, nsample * nsample);
      
    // b_nsxns = inv(b_nsxns)
    tracker_gpu_inverse<Dtype>(nsample, this->b_nseg_nseg_.mutable_gpu_data());
    
    tracker_saveMat<Dtype>("matrix_inv.bin", this->b_nseg_nseg_.cpu_data(), nsample, nsample * nsample);
    
    LOG(FATAL) << "Matrix inversion failed!";
}

INSTANTIATE_LAYER_GPU_FUNCS(LinearTrackerLayer);
}  // namespace caffe
