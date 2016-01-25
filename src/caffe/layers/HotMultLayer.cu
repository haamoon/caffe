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
#include "caffe/tracking_layers.hpp"
#include <cuda.h>
namespace caffe {
  
  __device__ double device_atomicAdd(double* address, double val)
  {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed,
                      __double_as_longlong(val +
                      __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
  }
  
  __device__ float device_atomicAdd(float* address, float val)
  {
    return atomicAdd(address, val);
  }
  
  //permute rows of matrix 'b' with respect to the values in array 'a'.  if from_ind_to_val == True: moves row i to a[i] and row a[i] to i otherwise
  // Let matrix A be the one-hot represenation of array 'a' (expanded in column direction) then:
  // if from_ind_to_value: C = A^\top x B  else: C = A x B
  template <typename Dtype>
  __global__ void HotRowPermute(const int nthreads, int N, const Dtype* a, 
                                                   const Dtype* a_lens, int a_len, const Dtype* b, 
                                                   int b_row, int b_col, Dtype* c, int c_row, int c_col, 
                                                   bool from_ind_to_value) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      //nthreads = N * a_len * c_col
      int tmp = index;
      int n = tmp % N;
      tmp /= N;
      int col = tmp % c_col;
      int i = tmp / c_col;
      
      int cur_a_len = (a_lens == NULL) ? a_len : a_lens[n];
      
      if(i < cur_a_len) { 
        const Dtype* a_pointer = a + n * a_len;
        const Dtype* b_pointer = b + n * b_row * b_col;
        Dtype* c_pointer = c + n * c_row * c_col;
        
        int from = from_ind_to_value ? i : a_pointer[i];
        int to = from_ind_to_value ? a_pointer[i] : i;
        device_atomicAdd(c_pointer + to * c_col + col, b_pointer[from * b_col + col]);
      }
    }
  }
   
  //permute columns of matrix 'b' with respect to the values in array 'a'.  If from_ind_to_val == True: moves column i to a[i] and moves column a[i] to i otherwise
  // Let matrix A be the one-hot represenation of array 'a' (expanded in row direction) then:
  // if(from_ind_to_value): C = B x A^\top else: C = B x A 
  // c_col is needed only in (from_ind_to_value == true) case
  template <typename Dtype>
  __global__ void HotColPermute(const int nthreads, int N, const Dtype* a, const Dtype* a_lens, int a_len, const Dtype* b, int b_row, int b_col, Dtype* c, int c_row, int c_col, bool from_ind_to_value) {
    
    CUDA_KERNEL_LOOP(index, nthreads) {
      //nthreads = N * a_len * c_row
      int tmp = index;
      int n = tmp % N;
      tmp /= N;
      int row = tmp % c_row;
      int i = tmp / c_row;
      
      
      int cur_a_len = (a_lens == NULL) ? a_len : a_lens[n];
      if(i < cur_a_len) {
        
        const Dtype* a_pointer = a + n * a_len;
        const Dtype* b_pointer = b + n * b_row * b_col;
        Dtype* c_pointer = c + n * c_row * c_col;
        
        int from = from_ind_to_value ? i : a_pointer[i];
        int to = from_ind_to_value ? a_pointer[i] : i;
         
        device_atomicAdd(c_pointer + c_col * row + to, b_pointer[b_col * row + from]);
      }
    }
  }

  template <typename Dtype>
  void HotMultLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, 
                                        const vector<Blob<Dtype>*>& top) {
    
    const Dtype* A_data = bottom[0]->gpu_data();
    const Dtype* B_data = bottom[1]->gpu_data();
    const Dtype* a_lens = NULL; 
    if(bottom.size() > 2) {
      a_lens = bottom[2]->gpu_data();
    }
    Dtype* C_data = top[0]->mutable_gpu_data();
    caffe_gpu_set(top[0]->count(), Dtype (0.0), C_data);
    if(row_mode_) {
      int nthreads = N_ * b_c_ * a_len_;
      //C = A x B
      HotRowPermute<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(nthreads, N_, A_data, a_lens, a_len_, B_data, b_r_, b_c_, C_data, a_len_, b_c_, false);
    } else {
      int nthreads = N_ * b_r_ * a_len_;
      //C = B x A
      HotColPermute<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(nthreads, N_, A_data, a_lens, a_len_, B_data, b_r_, b_c_, C_data, b_r_, a_len_, false);
    }  
  }
                  
  template <typename Dtype>
  void HotMultLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                         const vector<bool>& propagate_down, 
                                         const vector<Blob<Dtype>*>& bottom) {
    const Dtype* A_data = bottom[0]->gpu_data();
    const Dtype* C_diff = top[0]->gpu_diff();
    Dtype* B_diff = bottom[1]->mutable_gpu_diff();
    const Dtype* a_lens = NULL; 
    if(bottom.size() > 2) {
      a_lens = bottom[2]->gpu_data();
    }
    caffe_gpu_set(bottom[1]->count(), Dtype (0.0), B_diff);
    if(row_mode_) {
      int nthreads = N_ * b_c_ * a_len_;
      //B' = A^\top x C'
      HotRowPermute<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(nthreads, N_, A_data, a_lens, a_len_, C_diff, a_len_, b_c_, B_diff, b_r_, b_c_, true);
    } 
    else {
      int nthreads = N_ * b_r_ * a_len_;
      //B' = C' x A^\top
      HotColPermute<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(nthreads, N_, A_data, a_lens, a_len_, C_diff, b_r_, a_len_, B_diff, b_r_, b_c_, true);
    }                                       
  }
  
  INSTANTIATE_LAYER_GPU_FUNCS(HotMultLayer);
                                                                               
}  // namespace caffe
