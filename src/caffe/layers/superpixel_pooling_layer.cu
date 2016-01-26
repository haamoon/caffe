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
  __global__ void SuperpixelPoolingForward(const int nthreads,
                                           const Dtype* image_data, const Dtype* spixel_data, const Dtype* spixel_ptr,
                                           const Dtype* spixel_num, int image_width, int image_height,
                                           const Dtype* mask_size, int N, int spixel_ptr_len, int spixel_data_len,
                                           int channels, int crop_h_offset, int crop_w_offset, int input_height, int input_width, Dtype* top_data) {
    
    //nthreads = N_ * channels_ * spixel_num
    CUDA_KERNEL_LOOP(index, nthreads) {
      //To have a balance workload on each thread extract spixel last.
      int tmp = index;
      int n =  tmp % N;
      tmp /= N;
      int c = tmp % channels;
      int spixel = tmp / channels;
      
      Dtype sum = 0;
      if(spixel < spixel_num[n]) {
        const Dtype* cur_spixel_ptr = spixel_ptr + n * spixel_ptr_len + spixel;
        int start_ind = cur_spixel_ptr[0];
        int end_ind = cur_spixel_ptr[1];
        
        const Dtype* cur_mask_size = mask_size + 2 * n;
        Dtype h_ratio = image_height / cur_mask_size[0];
        Dtype w_ratio = image_width / cur_mask_size[1];
        const Dtype* cur_spixel_data = spixel_data + (n * spixel_data_len + start_ind) * 2;
        for(int i = start_ind; i < end_ind; i++) {
          int row = crop_h_offset + (int)(*(cur_spixel_data++) * h_ratio);
          int col = crop_w_offset + (int)(*(cur_spixel_data++) * w_ratio);
          sum += image_data[((n * channels + c ) * input_height + row) * input_width + col];
        }
        
        if(end_ind != start_ind) {
          sum /= (end_ind - start_ind);
        }
      }
      top_data[(n * (spixel_ptr_len - 1) + spixel) * channels + c] = sum;
    }
  }
  
  template <typename Dtype>
  __global__ void SuperpixelPoolingBackward(const int nthreads,
                                            const Dtype* top_diff, const Dtype* spixel_data, const Dtype* spixel_ptr,
                                            const Dtype* spixel_num, int image_width, int image_height,
                                            const Dtype* mask_size, int N, int spixel_ptr_len, int spixel_data_len,
                                            int channels, int crop_h_offset, int crop_w_offset, int input_height, int input_width,  Dtype* bottom_diff) {
    //nthreads = N_ * channels_
    CUDA_KERNEL_LOOP(index, nthreads) {
      int tmp = index;
      int n =  tmp % N;
      tmp /= N;
      int c = tmp % channels;
      
      const Dtype* cur_spixel_ptr = spixel_ptr + n * spixel_ptr_len;
      for(int spixel = 0; spixel < spixel_num[n]; spixel++) {
        int start_ind = cur_spixel_ptr[0];
        int end_ind = cur_spixel_ptr[1];
        
        const Dtype* cur_mask_size = mask_size + 2 * n;
        Dtype h_ratio = image_height / cur_mask_size[0];
        Dtype w_ratio = image_width / cur_mask_size[1];
        const Dtype* cur_spixel_data = spixel_data + (n * spixel_data_len + start_ind) * 2;
        for(int i = start_ind; i < end_ind; i++) {
          int row = crop_h_offset + (int)(*(cur_spixel_data++) * h_ratio);
          int col = crop_w_offset + (int)(*(cur_spixel_data++) * w_ratio);
          bottom_diff[((n * channels + c ) * input_height + row) *
                      input_width + col] +=
                      top_diff[(n * (spixel_ptr_len - 1) + spixel) *
                      channels + c] / (end_ind - start_ind);
        }
        cur_spixel_ptr++;
      }
    }
  }
  
//   template <typename Dtype>
//   __global__ void UnscaledSuperpixelPoolingBackward(const int nthreads,
//                                             const Dtype* top_diff, const Dtype* spixel_data, const Dtype* spixel_ptr,
//                                             const Dtype* spixel_num, int image_width, int image_height,
//                                             const Dtype* mask_size, int N, int spixel_ptr_len, int spixel_data_len,
//                                             int channels, Dtype* bottom_diff) {
//     //nthreads = N_ * channels_ * spixel_num
//     CUDA_KERNEL_LOOP(index, nthreads) {
//       int tmp = index;
//       int n =  tmp % N;
//       tmp /= N;
//       int c = tmp % channels;
//       int spixel = tmp / channels;
//       
//       if(spixel < spixel_num[n]) {
//         Dtype sum = 0;
//         spixel_ptr += n * spixel_ptr_len + spixel;
//         int start_ind = spixel_ptr[0];
//         int end_ind = spixel_ptr[1];
//         
//         spixel_data += (n * spixel_data_len + start_ind) * 2;
//         for(int i = start_ind; i < end_ind; i++) {
//           int row = *(spixel_data++);
//           int col = *(spixel_data++);
//           bottom_diff[((n * channels + c ) * image_height + row) *
//           image_width + col] +=
//           top_diff[(n * (spixel_ptr_len - 1) + spixel) *
//           channels + c] / (end_ind - start_ind);
//         }
//       }
//     }
//   }
                                            
  template <typename Dtype>
  void SuperpixelPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                                  const vector<Blob<Dtype>*>& top) {
    const Dtype* image_data = bottom[0]->cpu_data();
    const Dtype* spixel_data = bottom[1]->gpu_data();
    const Dtype* spixel_ptr = bottom[2]->gpu_data();
    const Dtype* spixel_num = bottom[3]->gpu_data();
    const Dtype* mask_size = bottom[4]->gpu_data();
    
    Dtype* top_data = top[0]->mutable_gpu_data();
    const int top_count = top[0]->count();
    
    SuperpixelPoolingForward<Dtype><<<CAFFE_GET_BLOCKS(top_count), CAFFE_CUDA_NUM_THREADS>>>(
                    top_count, image_data, spixel_data, spixel_ptr, spixel_num,
                    image_width_, image_height_, mask_size, N_, spixel_ptr_len_,
                    spixel_data_len_, channels_, crop_h_offset_, crop_w_offset_, input_height_, input_width_, top_data);
  }
  
  template <typename Dtype>
  void SuperpixelPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                                   const vector<bool>& propagate_down,
                                                   const vector<Blob<Dtype>*>& bottom) {
    
    CHECK(!propagate_down[1]) << "Can not backpropagate to spixel_data";
    CHECK(!propagate_down[2]) << "Can not backpropagate to spixel_ptr";
    CHECK(!propagate_down[3]) << "Can not backpropagate to spixel_num";
    CHECK(!propagate_down[4]) << "Can not backpropagate to mask_size";
    
    if (!propagate_down[0]) {
      return;
    }
    
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* spixel_data = bottom[1]->gpu_data();
    const Dtype* spixel_ptr = bottom[2]->gpu_data();
    const Dtype* spixel_num = bottom[3]->gpu_data();
    const Dtype* mask_size = bottom[4]->gpu_data();
    
    caffe_gpu_set(bottom[0]->count(), (Dtype)0.0, bottom_diff);
    
    int count = N_ * channels_;
    
    SuperpixelPoolingBackward<Dtype>
      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, spixel_data, spixel_ptr, spixel_num, image_width_,
      image_height_, mask_size, N_, spixel_ptr_len_, spixel_data_len_,
      channels_, crop_h_offset_, crop_w_offset_, input_height_, input_width_, bottom_diff);
  }
  
  INSTANTIATE_LAYER_GPU_FUNCS(SuperpixelPoolingLayer);
  
}  // namespace caffe
