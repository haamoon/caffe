#ifndef CAFFE_UTIL_TRACKER_MATH_H_
#define CAFFE_UTIL_TRACKER_MATH_H_

#include <stdint.h>

#include "glog/logging.h"

#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"

#include "magma.h"
#include "magma_lapack.h"

extern "C" {
  #include <clapack.h>
}
 
namespace caffe {
    
extern magma_int_t magma_init_val;

template <typename Dtype>
void tracker_cpu_inverse(const int n, Dtype* X);

template <typename Dtype>
void tracker_strided_add_scalar(const int N, const Dtype alpha, int incx, Dtype *X);


template <typename Dtype>
void tracker_printMat(std::ostream& buffer, const Dtype* mat, int col, int count);

template <typename Dtype>
void tracker_saveMat(string filename, const Dtype* mat, int col, int count);


template<typename Dtype>
void tracker_cpu_copy(const Dtype* A, const int ina, const int lda, 
                      Dtype* B, const int inb, const int ldb, 
                      const int nrows, const int ncols);


template<typename Dtype>
void tracker_cpu_gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M, 
                      const int N, const int K, const Dtype alpha, const Dtype* A, 
                      const int lda, const Dtype* B, const int ldb, const Dtype beta, Dtype* C);

template <typename Dtype>
void tracker_cpu_transpose(const int m, const int n, const Dtype* a, const int lda, Dtype* b, const int ldb);

template <typename Dtype>
void tracker_axpy(const int N, const Dtype alpha, const Dtype* X, Dtype* Y, const int ldx = 1, const int ldy = 1);

template<typename Dtype>
void tracker_cpu_csr_gemm(const CBLAS_TRANSPOSE TransA,
                        const CBLAS_TRANSPOSE TransB, const int M, const int N,
                        const int K, const Dtype alpha, const int nzz,
                        const Dtype* A, const Dtype* indices, const Dtype* ptr,
                        const Dtype* B, const Dtype beta, Dtype* C,
                        const CBLAS_ORDER orderC);

#ifndef CPU_ONLY  // GPU
template <typename Dtype>
void tracker_gpu_pos_inverse(int n, Dtype* X);


template <typename Dtype>
void tracker_gpu_inverse(int n, Dtype* X);

template <typename Dtype>
void tracker_gpu_clip_matrix(int a_rows, int a_cols, int clip_rows, int clip_cols, Dtype* a, int ldda, Dtype val);

template <typename Dtype>
void copy_gpu_matrix(int a_cols, int b_cols, const Dtype* a, int ldda, Dtype* b, int lddb);

template <typename Dtype>
void tracker_gpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A, const int lda, const Dtype* B, const int ldb, const Dtype beta,
    Dtype* C);
template <typename Dtype>
void tracker_gpu_strided_add_scalar(const int N, const Dtype alpha, int incx, Dtype* Y);

template <typename Dtype>
void tracker_gpu_transpose(const int m, const int n, const Dtype* a, const int lda, Dtype* b, const int ldb);

template<typename Dtype>
void tracker_gpu_csr_gemm(const CBLAS_TRANSPOSE TransA,
                          const CBLAS_TRANSPOSE TransB, const int M, const int N,
                          const int K, const Dtype alpha, int nzz, const Dtype* A,
                          const Dtype* indices, const Dtype* ptr, const Dtype* B,
                          const Dtype beta, Dtype* C, const CBLAS_ORDER orderC);
#endif  // !CPU_ONLY

}  // namespace caffe

#endif  // CAFFE_UTIL_MATH_FUNCTIONS_H_
