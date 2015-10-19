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
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/**
 * @brief MatMultLayer. Compute C = AxB it handles the case where either A or B is diagonal.
 */
template <typename Dtype>
class MatMultLayer : public Layer<Dtype> {
 public:
  explicit MatMultLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MinTopBlobs() const { return 1; }

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

  /// @brief The spatial dimensions of the first input.
  vector<int> a_shape_;
  /// @brief The spatial dimensions of the second input.
  vector<int> b_shape_;
  /// @brief The spatial dimensions of the output.
  vector<int> c_shape_;
 private:
};
}  // namespace caffe

#endif  // CAFFE_TRACKING_LAYERS_HPP_
