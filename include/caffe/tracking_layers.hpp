#ifndef CAFFE_LAYERS_LAYERS_HPP_
#define CAFFE_LAYERS_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/tracker_math.hpp"
#include <opencv2/core/core.hpp>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/power_layer.hpp"
#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/sequence_layers.hpp"
namespace caffe {
/**
 * @brief MatInvLayer. Compute A = (A + \lambda I)^{-1}.
 */
template <typename Dtype>
class MatInvLayer : public Layer<Dtype> {
public:
  virtual inline const char* type() const { return "MatInv"; }

  explicit MatInvLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  /// @brief The spatial dimensions of the col_buffer.
  vector<int> tmp_buffer_shape_;

private:
  float lambda_;
  vector<int> input_shape_;
  int N_;
  int dim_;
  int offset_;
  Blob<Dtype> tmp_buffer_; 
};

template <typename Dtype>
class SwitchLayer : public Layer<Dtype> {
public:
  explicit SwitchLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Switch"; }

  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int axis_;
  int outer_dim_;
  int selector_dim_;
  int inner_dim_;
};

/**
 * @brief MatMultLayer. Compute C = AxB it handles the case where either A or B is diagonal.
 */
template <typename Dtype>
class MatMultLayer : public Layer<Dtype> {
public:
  virtual inline const char* type() const { return "MatMult"; }
  explicit MatMultLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);
    
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // Compute height_out_ and width_out_ from other parameters.
  //  virtual void compute_output_shape() = 0;

private:
  /// @brief The spatial dimensions of the first input.
  vector<int> a_shape_;
  /// @brief The spatial dimensions of the second input.
  vector<int> b_shape_;

  int N_M_;
  int A_offset_;
  int B_offset_;
  int C_offset_;
  int D_1_;
  int D_2_;
  int D_3_;
  bool A_is_diag_;
  bool B_is_diag_;
  CBLAS_TRANSPOSE A_transpose_;
  CBLAS_TRANSPOSE B_transpose_;
};

/**
 * @brief MatTransposeLayer. Compute B = A.T.
 */
template <typename Dtype>
class MatTransposeLayer : public Layer<Dtype> {
public:
  virtual inline const char* type() const { return "MatTranspose"; }
  
  explicit MatTransposeLayer(const LayerParameter& param)
  : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);
  
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  
protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
private:
  int N_;
  int rows_;
  int cols_;
  int offset_;
};

template <typename Dtype>
class LinearTrackerLayer : public InnerProductLayer<Dtype> {
public:
  explicit LinearTrackerLayer(const LayerParameter& param)
  : InnerProductLayer<Dtype>(param) {}
  
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "LinearTracker"; }
  virtual inline int ExactNumBottomBlobs() const { return 5; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);
  
protected:
  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
   virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  void save_and_exit(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  
  int max_nseg_;
  float w_scale_;
  int feature_dim_;
  int sample_inter_;
  int max_ntrack_;
  int input_start_axis_;
  Dtype lambda_;
  //Blob<Dtype> b1_ntrack_nseg_;
  Blob<Dtype> b2_ntrack_nseg_;
  Blob<Dtype> b_nseg_nseg_;
  Blob<Dtype> o_;
};

template <typename Dtype>
class SegmentPoolingLayer : public Layer<Dtype> {
public:
  explicit SegmentPoolingLayer(const LayerParameter& param)
  : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);
  
  virtual inline const char* type() const { return "SegmentPooling"; }
  virtual inline int ExactNumBottomBlobs() const { return 5; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  
protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
private:
  int N_;
  int channels_;
  int input_ncell_;
  int max_data_len_;
  int nspatial_cell_;
  int max_nrows_;
  Blob<int> int_indices_;
  Blob<int> int_ptrs_;
};


template <typename Dtype>
class Filter2DLayer : public Layer<Dtype> {
 public:
  explicit Filter2DLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Filter2D"; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  string mode_;
  int input_rows_;
  int input_cols_;
};

/**
 * Normalization Layer
 * add by Binbin Xu
 * declanxu@gmail.com or declanxu@126.com
 */
template <typename Dtype>
class NormalizationLayer : public Layer<Dtype> {
public:
  explicit NormalizationLayer(const LayerParameter& param)
  : Layer<Dtype>(param) {}
  //virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
  //    const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);
  
  virtual inline const char* type() const {
    return "Normalization";
  }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  
protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  int N_;
  int D_;
  Blob<Dtype> sum_multiplier_;
  Blob<Dtype> norm_;
  Blob<Dtype> squared_;
};

/**
 * @brief Computes the L1 or L2 Loss, optionally on the L2 norm along channels
 *
 */

//Forward declare
template <typename Dtype> class ConvolutionLayer;
template <typename Dtype> class EltwiseLayer;

template <typename Dtype>
class L1LossLayer : public LossLayer<Dtype> {
public:
  explicit L1LossLayer(const LayerParameter& param)
  : LossLayer<Dtype>(param), sign_() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);    
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);
  
  virtual inline const char* type() const { return "L1Loss"; }
  
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }
  
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  
protected:
  /// @copydoc L1LossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  Blob<Dtype> sign_, mask_, plateau_l2_;
  float scale_;
  Dtype normalize_coeff_;
  
  // Extra layers to do the dirty work using already implemented stuff
  shared_ptr<EltwiseLayer<Dtype> > diff_layer_;
  Blob<Dtype> diff_;
  vector<Blob<Dtype>*> diff_top_vec_;
  shared_ptr<PowerLayer<Dtype> > square_layer_;
  Blob<Dtype> square_output_;
  vector<Blob<Dtype>*> square_top_vec_;
  shared_ptr<ConvolutionLayer<Dtype> > sum_layer_;
  Blob<Dtype> sum_output_;
  vector<Blob<Dtype>*> sum_top_vec_;
  shared_ptr<PowerLayer<Dtype> > sqrt_layer_;
  Blob<Dtype> sqrt_output_;
  vector<Blob<Dtype>*> sqrt_top_vec_;
};

template <typename Dtype>
    class RecurrentTrackerLayer : public RecurrentLayer<Dtype> {
  public:
    explicit RecurrentTrackerLayer(const LayerParameter& param)
        : RecurrentLayer<Dtype>(param) {}
        
    virtual inline const char* type() const { return "RecurrentTracker"; }
    virtual inline int MinBottomBlobs() const { return 3; }
    virtual inline int MaxBottomBlobs() const { return 3; }
    virtual inline int ExactNumTopBlobs() const { return 1; }
  protected:
    virtual void FillUnrolledNet(NetParameter* net_param) const;
    virtual void RecurrentInputBlobNames(vector<string>* names) const;
    virtual void RecurrentOutputBlobNames(vector<string>* names) const;
    virtual void RecurrentInputShapes(vector<BlobShape>* shapes) const;
    virtual void OutputBlobNames(vector<string>* names) const;
    virtual void InputBlobNames(vector<string>* names) const;
};

}  // namespace caffe

#endif  // CAFFE_TRACKING_LAYERS_HPP_
