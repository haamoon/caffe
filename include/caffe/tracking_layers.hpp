#ifndef CAFFE_LAYERS_LAYERS_HPP_
#define CAFFE_LAYERS_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/sequence_layers.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {
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

  virtual inline const char* type() const { return "Switch"; }
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
 protected:

  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  private:
  int D_1_;
  int D_2_;
  int input_offset_;
  int N_;
};

template <typename Dtype>
    class TrackerLayer : public RecurrentLayer<Dtype> {
  public:
    explicit TrackerLayer(const LayerParameter& param)
        : RecurrentLayer<Dtype>(param) {}
        
    virtual inline const char* type() const { return "Tracker"; }  
  protected:
    virtual void FillUnrolledNet(NetParameter* net_param) const;
    virtual void RecurrentInputBlobNames(vector<string>* names) const;
    virtual void RecurrentOutputBlobNames(vector<string>* names) const;
    virtual void RecurrentInputShapes(vector<BlobShape>* shapes) const;
    virtual void OutputBlobNames(vector<string>* names) const;
};

/**
 * @brief Pools the input image with respect to the segmentation mask by taking the average within segments.
 *
 */
 
template <typename Dtype>
class MaskedPoolingLayer : public Layer<Dtype> {
 public:
  explicit MaskedPoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MaskedPooling"; }
  virtual inline int ExactNumBottomBlobs() const { return 4; }
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
  	int max_nseg_;
  	int channels_;
  	int mask_lenght_;
};


/**
 * @brief Pools the input image with respect to the non-overlapping superpixels by taking the average within each superpixel.
 *
 */
 
template <typename Dtype>
class SuperpixelPoolingLayer : public Layer<Dtype> {
 public:
  explicit SuperpixelPoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SuperpixelPooling"; }
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
  	int image_height_;
	int image_width_;
	
  	int spixel_data_len_;
  	int spixel_ptr_len_;
};



/**
 * @brief Pools the input rows of with respect to the overlapping set of rows by taking the average.
 *
 */
 
template <typename Dtype>
class RowPoolingLayer : public Layer<Dtype> {
 public:
  explicit RowPoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RowPooling"; }
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
  	int ncol_;
  	int nrow_;
  	int seg_data_len_;
  	int seg_ptr_len_;
};

/**
 * @brief compute the matching segment to each track and assigns values to the 
 * label vector V with respect to the overlaps of the each segment to each track
 *
 */
 
template <typename Dtype>
class TrackerMatchingLayer : public Layer<Dtype> {
 public:
  explicit TrackerMatchingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "TrackerMatching"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  Blob<int> max_indeces_; 
  int N_;
  int max_nseg_;
  int max_ntrack_;
};


}  // namespace caffe

#endif  // CAFFE_TRACKING_LAYERS_HPP_
