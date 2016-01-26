#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/tracking_layers.hpp"

namespace caffe {
  
  template <typename Dtype>
  void SuperpixelPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                 const vector<Blob<Dtype>*>& top) {
    
    CHECK_GE(bottom[0]->num_axes(), 4) << "Input(1) image must have at least 4 axes, "
    << "corresponding to (..., channels, height, width)";
    
    CHECK_GE(bottom[1]->num_axes(), 3) << "Input(2) spixel_data must have 3 axes, "
    << "corresponding to (..., pixel_id, row_num = 0/column_num = 1)";
    CHECK_GE(bottom[2]->num_axes(), 2) << "Input(3) spixel_ptr must have 2 axes, "
    << "corresponding to (..., ptr)";
    CHECK_GE(bottom[3]->num_axes(), 1) << "Input(4) spixel_num must have 1 axes";
    CHECK_GE(bottom[4]->num_axes(), 2) << "Input(5) mask_size must have 2 axes, "
    << "corresponding to (..., height = 0/width = 1)";
    
    
    if(bottom.size() > 5) {
      CHECK_GE(bottom[5]->num_axes(), 3) << "Input(1) image must have at least 3 axes, "
      << "corresponding to (..., height, width)";
      // Construct a map from top blobs to layer inds, skipping over in-place
      // connections.
      map<Blob<Dtype>*, int> down_map;
      for (int layer_ind = 0; layer_ind < this->net_->top_vecs().size();
           ++layer_ind) {
        vector<Blob<Dtype>*> tops = this->net_->top_vecs()[layer_ind];
        for (int top_ind = 0; top_ind < tops.size(); ++top_ind) {
          if (down_map.find(tops[top_ind]) == down_map.end()) {
            down_map[tops[top_ind]] = layer_ind;
          }
        }
      }
      // Walk back from the first bottom, keeping track of all the blobs we pass.
      set<Blob<Dtype>*> path_blobs;
      Blob<Dtype>* blob = bottom[0];
      int layer_ind;
      // TODO this logic can be simplified if all blobs are tops
      path_blobs.insert(blob);
      while (down_map.find(blob) != down_map.end()) {
        layer_ind = down_map[blob];
        if (this->net_->bottom_vecs()[layer_ind].size() == 0) {
          break;
        }
        blob = this->net_->bottom_vecs()[layer_ind][0];
        path_blobs.insert(blob);
      }
      // Now walk back from the fifth bottom, until we find a blob of intersection.
      Blob<Dtype>* inter_blob = bottom[5];
      while (path_blobs.find(inter_blob) == path_blobs.end()) {
        CHECK(down_map.find(inter_blob) != down_map.end())
        << "Cannot align apparently disconnected blobs.";
        layer_ind = down_map[inter_blob];
        CHECK_GT(this->net_->bottom_vecs()[layer_ind].size(), 0)
        << "Cannot align apparently disconnected blobs.";
        inter_blob = this->net_->bottom_vecs()[layer_ind][0];
      }
      // Compute the coord map from the blob of intersection to each bottom.
      vector<DiagonalAffineMap<Dtype> > coord_maps(2,
                                                   DiagonalAffineMap<Dtype>::identity(2));
      //   std::stringstream buffer;
         for (int i = 0; i < 2; ++i) {
           int index = (i == 0) ? 0 : 5;
      //     buffer << "i = " << i << std::endl;
           for (Blob<Dtype>* blob = bottom[index]; blob != inter_blob;
                blob = this->net_->bottom_vecs()[down_map[blob]][0]) {
             shared_ptr<Layer<Dtype> > layer = this->net_->layers()[down_map[blob]];
             coord_maps[i] = coord_maps[i].compose(layer->coord_map());
      
       //      buffer << layer->layer_param().name() << ": ";
       //      buffer << layer->coord_map().coefs()[0].first << ", " << layer->coord_map().coefs()[0].second << std::endl;
           }
         }
      //   LOG(ERROR) << buffer.str();
      // Compute the mapping from first bottom coordinates to second.
      DiagonalAffineMap<Dtype> crop_map =
      coord_maps[1].compose(coord_maps[0].inv());
      for (int i = 0; i < 2; ++i) {
        // Check for scale mismatch (unfortunately, CHECK_DOUBLE_EQ does not
        // support a message like the other CHECKs).
        //CHECK_DOUBLE_EQ(crop_map.coefs()[i].first, 1);
        CHECK_LE(crop_map.coefs()[i].second, 0) << "Negative crop width.";
        // Check that the crop width is an integer.
        CHECK_DOUBLE_EQ(crop_map.coefs()[i].second,
                        round(crop_map.coefs()[i].second));
      }
      
      crop_h_offset_ = - round(crop_map.coefs()[0].second);
      crop_w_offset_ = - round(crop_map.coefs()[1].second);
      crop_h_scale_ = crop_map.coefs()[0].first; 
      crop_w_scale_ = crop_map.coefs()[1].first;
    }
    else {
      crop_h_offset_ = 0;
      crop_w_offset_ = 0;
      crop_h_scale_ = static_cast<Dtype>(1);
      crop_w_scale_ = static_cast<Dtype>(1);
    }
  }
  
  template <typename Dtype>
  void SuperpixelPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                              const vector<Blob<Dtype>*>& top) {
    int input_start_axis = bottom[0]->CanonicalAxisIndex(-3);
    N_ = bottom[0]->count(0, input_start_axis);
    CHECK_EQ(N_, bottom[1]->count(0, bottom[1]->CanonicalAxisIndex(-2)));
    CHECK_EQ(N_, bottom[2]->count(0, bottom[2]->CanonicalAxisIndex(-1)));
    CHECK_EQ(N_, bottom[3]->count(0, bottom[3]->CanonicalAxisIndex(-1) + 1));
    CHECK_EQ(N_, bottom[4]->count(0, bottom[4]->CanonicalAxisIndex(-1)));
    
    channels_ = bottom[0]->shape(input_start_axis);
    
    
    input_height_ = bottom[0]->shape(input_start_axis + 1);
    input_width_ = bottom[0]->shape(input_start_axis + 2);
    
    Dtype image_height = (crop_h_scale_ * input_height_);
    Dtype image_width = (crop_w_scale_ * input_width_);
    
    CHECK_DOUBLE_EQ(image_height, round(image_height));
    CHECK_DOUBLE_EQ(image_width, round(image_width));
    
    image_height_ = round(image_height);
    image_width_ = round(image_width);
    
    //bottom[1] has size ... x spixel_data_len_ x 2
    spixel_data_len_ = bottom[1]->shape(bottom[1]->CanonicalAxisIndex(-2));
    
    //bottom[2] has segmens starting indeces with size ... x spixel_ptr_len_
    spixel_ptr_len_ = bottom[2]->shape(bottom[2]->CanonicalAxisIndex(-1));
    
    //X_t = top is a (spixel_ptr_len_ - 1) x channels_ matrix
    // We need s+1 pointer to show start,end pairs of spixels in spixel_data
    vector<int> top_shape = bottom[3]->shape();
    top_shape.push_back(spixel_ptr_len_ - 1);
    top_shape.push_back(channels_);
    
    top[0]->Reshape(top_shape);
  }
  
  template <typename Dtype>
  void SuperpixelPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                                  const vector<Blob<Dtype>*>& top) {
    
    const Dtype* image_data = bottom[0]->cpu_data();
    const Dtype* spixel_data = bottom[1]->cpu_data();
    const Dtype* spixel_ptr = bottom[2]->cpu_data();
    const Dtype* spixel_num = bottom[3]->cpu_data();
    const Dtype* mask_size = bottom[4]->cpu_data();
    
    Dtype* top_data = top[0]->mutable_cpu_data();
    const int top_count = top[0]->count();
    
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = 0;
    }
    
    // The main loop
    for (int n = 0; n < N_; ++n) {
      Dtype h_ratio = image_height_ / mask_size[0];
      Dtype w_ratio = image_width_ / mask_size[1];
      for (int c = 0; c < channels_; ++c) {
        //iterate over superpixels
        for(int spixel = 0; spixel < spixel_num[n]; ++spixel) {
          CHECK_LT(spixel_num[n], spixel_ptr_len_) << "Number of superpixels" << 
                  " exceeds the maximum lenght of the ptr";
          int start_ind = spixel_ptr[spixel];
          int end_ind = spixel_ptr[spixel + 1];

          for(int i = start_ind; i < end_ind; i++) {
            int row = crop_h_offset_ + (int)(spixel_data[i * 2] * h_ratio);
            int col = crop_w_offset_ + (int)(spixel_data[i * 2 + 1] * w_ratio);
            top_data[spixel * channels_ + c] += image_data[row * input_width_ + col]/(end_ind - start_ind);
          }
        }
        image_data += bottom[0]->count(bottom[0]->CanonicalAxisIndex(-2));
      }
      top_data += channels_ * (spixel_ptr_len_ - 1);
      spixel_ptr += spixel_ptr_len_;
      spixel_data += spixel_data_len_ * 2;
      mask_size += 2;
    }
  }
  
  template <typename Dtype>
  void SuperpixelPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                   const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    CHECK(!propagate_down[1]) << "Can not backpropagate to spixel_data";
    CHECK(!propagate_down[2]) << "Can not backpropagate to spixel_ptr";
    CHECK(!propagate_down[3]) << "Can not backpropagate to spixel_num";
    CHECK(!propagate_down[4]) << "Can not backpropagate to mask_size";
    
    if (!propagate_down[0]) {
      return;
    }
    
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* spixel_data = bottom[1]->cpu_data();
    const Dtype* spixel_ptr = bottom[2]->cpu_data();
    const Dtype* spixel_num = bottom[3]->cpu_data();
    const Dtype* mask_size = bottom[4]->cpu_data();
    
    for (int i = 0; i < bottom[0]->count(); ++i) {
      bottom_diff[i] = 0;
    }
    
    // The main loop
    for (int n = 0; n < N_; ++n) {
      Dtype h_ratio = image_height_ / mask_size[0];
      Dtype w_ratio = image_width_ / mask_size[1];
      for (int c = 0; c < channels_; ++c) {
        //iterate over superpixels
        for(int spixel = 0; spixel < spixel_num[n]; ++spixel) {
          int start_ind = spixel_ptr[spixel];
          int end_ind = spixel_ptr[spixel + 1];
          CHECK_GE(end_ind, start_ind);
          for(int i = start_ind; i < end_ind; i++) {
            int row = crop_h_offset_ + (int)(spixel_data[i * 2] * h_ratio);
            int col = crop_w_offset_ + (int)(spixel_data[i * 2 + 1] * w_ratio);
            bottom_diff[row * input_width_ + col] +=
              top_diff[spixel * channels_ + c]/(end_ind - start_ind);
          }
        }
        bottom_diff += bottom[0]->count(bottom[0]->CanonicalAxisIndex(-2));
      }
      top_diff += channels_ * (spixel_ptr_len_ - 1);  	
      spixel_ptr += spixel_ptr_len_;
      spixel_data += spixel_data_len_ * 2;
      mask_size += 2;
    }
  }
  
  
#ifdef CPU_ONLY
  STUB_GPU(SuperpixelPoolingLayer);
#endif
  
  INSTANTIATE_CLASS(SuperpixelPoolingLayer);
  REGISTER_LAYER_CLASS(SuperpixelPooling);
  
}  // namespace caffe
