#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/tracker_math.hpp"
//#include "magma_internal.h"
//#include "batched_kernel_param.h"
#include "cublas_v2.h"
#define THREADS_PER_BLOCK_CSR 32

namespace caffe {

magma_int_t magma_init_val  = magma_init();

template <>
void tracker_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const int lda, const float* B, const int ldb, const float beta,
    float* C) {
  // Note that cublas follows fortran order.
  //int lda = (TransA == CblasNoTrans) ? K : M;
  //int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void tracker_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const int lda, const double* B, const int ldb, const double beta,
    double* C) {
  // Note that cublas follows fortran order.
  //int lda = (TransA == CblasNoTrans) ? K : M;
  //int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void tracker_gpu_pos_inverse<float>(int n, float* X)
{
    magma_int_t info;
    magma_spotrf_gpu(MagmaLower, n, X, n, &info);
    if (info != 0) {
      LOG(FATAL) << "magma_spotrf_gpu returned error " << (int) info << ":" << magma_strerror( info );
    }
    magma_spotri_gpu(MagmaLower, n, X, n, &info);
    if (info != 0) {
      LOG(FATAL) << "magma_spotri_gpu returned error " << (int) info << ":" << magma_strerror( info );
    }
    magmablas_ssymmetrize(MagmaLower, n, X, n);
}

template <>
void tracker_gpu_pos_inverse<double>(int n, double* X)
{
  magma_int_t info;
  magma_dpotrf_gpu(MagmaLower, n, X, n, &info);
  if (info != 0) {
    LOG(FATAL) << "magma_dpotrf_gpu returned error " << (int) info << ":" << magma_strerror( info );
  }
  magma_dpotri_gpu(MagmaLower, n, X, n, &info);
  if (info != 0) {
    LOG(FATAL) << "magma_dpotri_gpu returned error " << (int) info << ":" << magma_strerror( info );
  }
  magmablas_dsymmetrize(MagmaLower, n, X, n);
}

template <>
void tracker_gpu_inverse<double>(int n, double* X)
{
  magma_int_t ldwork; // size of dwork
  magma_int_t *piv , info; // piv - array of indices of int
  double *dwork; 
  magma_int_t err;
  ldwork = n * magma_get_sgetri_nb (n); // workspace size
  err = magma_dmalloc ( &dwork , ldwork ); // dev. mem. for ldwork
  piv =( magma_int_t *) malloc (n * sizeof ( magma_int_t )); // host mem.
  
  magma_dgetrf_gpu(n, n, X, n, piv, &info);
  if (info != 0) {
    LOG(FATAL) << "magma_dpotrf_gpu returned error " << (int) info << ":" << magma_strerror( info );
  }
  magma_dgetri_gpu(n, X, n, piv, dwork, ldwork, &info);
  if (info != 0) {
    LOG(FATAL) << "magma_dpotri_gpu returned error " << (int) info << ":" << magma_strerror( info );
  }
  magma_free( dwork );
  free ( piv ); // free host memory
}

template <>
void tracker_gpu_inverse<float>(int n, float* X)
{
  double* Xd;
  magma_int_t info;
  
  magma_dmalloc( &Xd , n * n);
  magmablas_slag2d(n, n, X, n, Xd, n, &info);
  if (info != 0) {
    LOG(FATAL) << "magmablas_slag2d returned error " << (int) info << ":" << magma_strerror( info );
  }
  tracker_gpu_inverse<double>(n, Xd);
  
  magmablas_dlag2s(n,n,Xd,n,X,n,&info);
  if (info != 0) {
    LOG(FATAL) << "magmablas_dlag2s returned error " << (int) info << ":" << magma_strerror( info );
  }
  magma_free( Xd );
  
//   magma_int_t ldwork; // size of dwork
//   magma_int_t *piv , info; // piv - array of indices of int
//   float *dwork; 
//   magma_int_t err;
//   ldwork = n * magma_get_sgetri_nb (n); // workspace size
//   err = magma_smalloc ( &dwork , ldwork ); // dev. mem. for ldwork
//   piv =( magma_int_t *) malloc (n * sizeof ( magma_int_t )); // host mem.
//   
//   magma_sgetrf_gpu(n, n, X, n, piv, &info);
//   if (info != 0) {
//     LOG(FATAL) << "magma_dpotrf_gpu returned error " << (int) info << ":" << magma_strerror( info );
//   }
//   magma_sgetri_gpu(n, X, n, piv, dwork, ldwork, &info);
//   if (info != 0) {
//     LOG(FATAL) << "magma_dpotri_gpu returned error " << (int) info << ":" << magma_strerror( info );
//   }
//   
//   magma_free( dwork );
//   free ( piv ); // free host memory
}




//  template <>
//  void tracker_gpu_cholesky<float>(int batch_size, int n, const float* A, float* cholesky) {
//    magma_init();
//    float **dA_array = NULL:   
//    magma_queue_t queue = NULL;
//    magma_queue_create( &queue );
//    
//    magma_int_t* d_info;
//    magma_int_t* cpu_info;
//    magma_int_t info;
//    
//    CHECK_EQ(magma_malloc((void**) &dA_array, (batch_size)*sizeof(float*))), MAGMA_SUCCESS);
//    CHECK_EQ(magma_malloc_cpu((void**) &cpu_info, (size)*sizeof(magma_int_t))), MAGMA_SUCCESS);
//    CHECK_EQ(magma_malloc((void**) &d_info, (batch_size)*sizeof(magma_int_t))), MAGMA_SUCCESS);
//    
//    magma_scopy(n * n * batch_size, A, 1, cholesky, 1);
//    magma_sset_pointer( dA_array, cholesky, n, 0, 0, n * n, batch_size, queue);
//    info = magma_spotrf_batched(MagmaLower, n, dA_array, n, d_info, batch_size, queue);
//    
//    
//    magma_getvector( batch_size, sizeof(magma_int_t), d_info, 1, cpu_info, 1);
//    for (int i = 0; i < batch_size; i++) {
//       if (cpu_info[i] != 0 ) {
//          LOG(FATAL) << "magma_spotrf_batched matrix " << i << " returned diag error " << (int)cpu_info[i];
//       }
//    }
//    if (info != 0) {
//       LOG(FATAL) << "magma_spotrf_batched returned argument error " << (int) info << " " << magma_strerror( info );
//    }             
//             
//    magma_free(dA_array);
//    magma_free(d_info);
//    magma_free_cpu(cpu_info);
//    magma_queue_destroy(queue);
//  }
//  
//   template <>
//   void tracker_gpu_spo_linear<float>(int batch_size, int n, int m, const float* cholesky, const float * B, float* X) {
//     magma_init();
//     float **dcholesky_array = NULL:   
//     magma_queue_t queue = NULL;
//     magma_queue_create( &queue );
//     
//     magma_int_t info;
//     magma_sset_pointer( dA_array, cholesky, n, 0, 0, n * n, batch_size, queue);
//     CHECK_EQ(magma_malloc((void**) &dcholesky_array, (batch_size)*sizeof(float*))), MAGMA_SUCCESS);
//  
//     
//     float **dW1_displ  = NULL;
//     float **dW2_displ  = NULL;
//     float **dW3_displ  = NULL;
//     float **dW4_displ  = NULL;
//     float **dinvA_array = NULL;
//     float **dwork_array = NULL;
// 
//     magma_malloc((void**)&dW1_displ,  batchCount * sizeof(*dW1_displ));
//     magma_malloc((void**)&dW2_displ,  batchCount * sizeof(*dW2_displ));
//     magma_malloc((void**)&dW3_displ,  batchCount * sizeof(*dW3_displ));
//     magma_malloc((void**)&dW4_displ,  batchCount * sizeof(*dW4_displ));
//     magma_malloc((void**)&dinvA_array, batchCount * sizeof(*dinvA_array));
//     magma_malloc((void**)&dwork_array, batchCount * sizeof(*dwork_array));
// 
//     magma_int_t invA_msize = magma_roundup( n, TRI_NB )*TRI_NB;
//     magma_int_t dwork_msize = n*nrhs;
//     float* dinvA      = NULL;
//     float* dwork      = NULL; // dinvA and dwork are workspace in strsm
//     magma_smalloc( &dinvA, invA_msize * batchCount);
//     magma_smalloc( &dwork, dwork_msize * batchCount );
// 
//     if ( dW1_displ == NULL || dW2_displ == NULL || dW3_displ   == NULL || dW4_displ   == NULL || 
//          dinvA_array == NULL || dwork_array == NULL || dinvA     == NULL || dwork     == NULL ) {
//       LOG(FATAL) << "magma_spotrs_batched returned argument error ";
//     }
//     
//     magmablas_slaset_q( MagmaFull, invA_msize, batchCount, MAGMA_S_ZERO, MAGMA_S_ZERO, dinvA, invA_msize, queue );
//     magmablas_slaset_q( MagmaFull, dwork_msize, batchCount, MAGMA_S_ZERO, MAGMA_S_ZERO, dwork, dwork_msize, queue );
//     magma_sset_pointer( dwork_array, dwork, n, 0, 0, dwork_msize, batchCount, queue );
//     magma_sset_pointer( dinvA_array, dinvA, TRI_NB, 0, 0, invA_msize, batchCount, queue );
//     
//     info = magma_spotrs_batched(MagmaLower, n, m, dcholesky_array, n, dB_array, , batch_size, queue);              
//     
//     if (info != 0) {
//        LOG(FATAL) << "magma_spotrs_batched returned argument error " << (int) info << " " << magma_strerror( info );
//     }
//     
//     // A = L L^T
//     // solve LX= B ==> dwork = L^{-1} B
//     magmablas_strsm_outofplace_batched( MagmaLeft, MagmaLower, MagmaNoTrans, MagmaNonUnit, 1,
//                     n, nrhs,
//                     c_one,
//                     dA_array,       ldda, // dA
//                     dB_array,      lddb, // dB
//                     dwork_array,        n, // dX //output
//                     dinvA_array,  invA_msize, 
//                     dW1_displ,   dW2_displ, 
//                     dW3_displ,   dW4_displ,
//                     1, batchCount, queue );
// 
//    // solve L^{T}X= dwork ==> X = L^{-T} dwork
//    magmablas_strsm_outofplace_batched( MagmaLeft, MagmaLower, MagmaConjTrans, MagmaNonUnit, 1,
//                     n, nrhs,
//                     c_one,
//                     dA_array,       ldda, // dA
//                     dwork_array,        n, // dB 
//                     dB_array,   lddb, // dX //output
//                     dinvA_array,  invA_msize, 
//                     dW1_displ,   dW2_displ, 
//                     dW3_displ,   dW4_displ,
//                     1, batchCount, queue );
//                     
//                     
//     magma_queue_sync(queue);
// 
//     magma_free(dW1_displ);
//     magma_free(dW2_displ);
//     magma_free(dW3_displ);
//     magma_free(dW4_displ);
//     magma_free(dinvA_array);
//     magma_free(dwork_array);
//     magma_free( dinvA );
//     magma_free( dwork );   
//     magma_queue_destroy(queue);
//  }

template <>
void copy_gpu_matrix<float>(int a_rows, int a_cols, const float* a, int ldda, float* b, int lddb) {
  //LOG(ERROR) << a_rows << ", " << a_cols << ", " << ldda << ", " << lddb;
  if(a_rows == 0 || a_cols == 0 || ldda == 0 || lddb == 0) 
    return;
  magmablas_slacpy(MagmaFull, a_cols, a_rows, a, ldda, b, lddb);
}


template <>
void tracker_gpu_clip_matrix<float>(int a_rows, int a_cols, int clip_rows, int clip_cols, float* a, int ldda, float val) {
  
  /* Set extra rows to zero
   * xxxxxxxxxxxxxxx
   * xxxxxxxxxxxxxxx
   * xxxxxxxxxxxxxxx
   * 000000000000000
   * 000000000000000
   * 000000000000000
  */
  if(a_rows > clip_rows) {
    magmablas_slaset(MagmaFull, a_rows - clip_rows, a_cols, val, val,a + clip_rows * ldda,ldda);
  }
  
  /* Set extra columns to zero
   * xxxxxxxxxx00000
   * xxxxxxxxxx00000
   * xxxxxxxxxx00000
   * 000000000000000
   * 000000000000000
   * 000000000000000
   */
  if(a_cols > clip_cols) {
    magmablas_slaset(MagmaFull, clip_rows, a_cols - clip_cols, val, val, a + clip_cols, ldda);
  }
}


template <>
void tracker_gpu_clip_matrix<double>(int a_rows, int a_cols, int clip_rows, int clip_cols, double* a, int ldda, double val) {
  if(a_rows > clip_rows) {
    magmablas_dlaset(MagmaFull, a_rows - clip_rows, a_cols, val, val,a + clip_rows * ldda,ldda);
  }
  if(a_cols > clip_cols) {
    magmablas_dlaset(MagmaFull, clip_rows, a_cols - clip_cols, val, val, a + clip_cols, ldda);
  }
}

template <>
void copy_gpu_matrix<double>(int a_rows, int a_cols, const double* a, int ldda, double* b, int lddb) {
  if(a_rows == 0 || a_cols == 0 || ldda == 0 || lddb == 0) 
    return;
  magmablas_dlacpy(MagmaFull, a_cols, a_rows, a, ldda, b, lddb);
}

template <typename Dtype>
__global__ void kernel_strided_add_scalar(const int nthreads, int incx, Dtype alpha, Dtype* data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    data[index * incx] += alpha;
  }
}

template <>
void tracker_gpu_strided_add_scalar(const int N, const float alpha, int incx, float* Y) {
  int nthreads = ceil(((float)N) / incx);
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_strided_add_scalar<float><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      nthreads, incx, alpha, Y);
}

template <>
void tracker_gpu_strided_add_scalar(const int N, const double alpha, int incx, double* Y) {
  int nthreads = N / incx;
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_strided_add_scalar<double><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      nthreads, incx, alpha, Y);
}

template <typename Dtype>
__global__ void kernel_transpose(const int nthreads, const int m, const int n, const Dtype* a, const int lda, Dtype* b, const int ldb) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int row = index / n;
    const int col = index % n;
    const int from = row * lda + col;
    const int to = col * ldb + row;
    
    if(a == b && col > row) 
    {
        const Dtype tmp = b[from];
        b[from] = b[to];
        b[to] = tmp;
    } else {
      b[to] = a[from];
    }
  }
}

template <typename Dtype>
void tracker_gpu_transpose(const int m, const int n, const Dtype* a, const int lda, Dtype* b, const int ldb) {
  if(a == b && (lda != n || ldb != m)) {
    LOG(FATAL) << "For inplace transpose data has to be contiguous in memory";
  }
  int nthreads = m * n;
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_transpose<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
    nthreads, m, n, a, lda, b, ldb);
}

template void tracker_gpu_transpose<float>(const int m, const int n, const float* a, const int lda, float* b, const int ldb);
template void tracker_gpu_transpose<double>(const int m, const int n, const double* a, const int lda, double* b, const int ldb);



  
template<typename Dtype>
__device__ void tracker_gpu_csr_gemm_kernel_core(const int M, const int N,
                                               const int K, const Dtype alpha,
                                               int nzz, const Dtype* A,
                                               const Dtype* indices,
                                               const Dtype* ptr, const Dtype* B,
                                               const int ldb1, const int ldb2,
                                               const Dtype beta, Dtype* C,
                                               const int ldc1, const int ldc2) {
  __shared__ volatile Dtype sums[THREADS_PER_BLOCK_CSR * 2];

  for (int rowA = blockIdx.x; rowA < M; rowA += gridDim.x) {
    const int begin = (int)ptr[rowA];
    const int end = (int)ptr[rowA + 1];
    const int offset_c_part = rowA * ldc1;
    for (int colC = blockIdx.y; colC < N; colC += gridDim.y) {
      Dtype sum = 0.0;
      const int offset_b_part = colC * ldb2;
      for (int pos = begin + threadIdx.x; pos < end; pos +=
          THREADS_PER_BLOCK_CSR) {
        const int colA = (int)indices[pos];
        sum += A[pos] * B[colA * ldb1 + offset_b_part];
      }
      sums[threadIdx.x] = sum;
      __syncthreads();

      /* hardcoded reduction for 32 threads */
      sums[threadIdx.x] += sums[threadIdx.x + 16];
      sums[threadIdx.x] += sums[threadIdx.x + 8];
      sums[threadIdx.x] += sums[threadIdx.x + 4];
      sums[threadIdx.x] += sums[threadIdx.x + 2];
      sums[threadIdx.x] += sums[threadIdx.x + 1];

      if (threadIdx.x == 0) {
        const int offsetC = offset_c_part + colC * ldc2;
        C[offsetC] = beta * C[offsetC] + alpha * sums[0];
      }
    }
  }
}

template<typename Dtype>
__global__ void tracker_gpu_csr_gemm_kernel(const CBLAS_TRANSPOSE TransB,
                                          const int M, const int N, const int K,
                                          const Dtype alpha, int nzz,
                                          const Dtype* A, const Dtype* indices,
                                          const Dtype* ptr, const Dtype* B,
                                          const Dtype beta, Dtype* C,
                                          const CBLAS_ORDER orderC) {
  if (orderC == CblasRowMajor) {
    if (TransB == CblasNoTrans) {
      tracker_gpu_csr_gemm_kernel_core(M, N, K, alpha, nzz, A, indices, ptr, B, N,
                                     1, beta, C, N, 1);
    } else {
      tracker_gpu_csr_gemm_kernel_core(M, N, K, alpha, nzz, A, indices, ptr, B, 1,
                                     K, beta, C, N, 1);
    }
  } else {
    if (TransB == CblasNoTrans) {
      tracker_gpu_csr_gemm_kernel_core(M, N, K, alpha, nzz, A, indices, ptr, B, N,
                                     1, beta, C, 1, M);
    } else {
      tracker_gpu_csr_gemm_kernel_core(M, N, K, alpha, nzz, A, indices, ptr, B, 1,
                                     K, beta, C, 1, M);
    }
  }
}

template<typename Dtype>
__device__ void tracker_gpu_csr_rank1_update_kernel_core(const int M, const int N,
                                                       const Dtype alpha,
                                                       const Dtype* A,
                                                       const Dtype* indices,
                                                       const Dtype* ptr,
                                                       const Dtype* B, int ldb,
                                                       Dtype* C, const int ldc1,
                                                       const int ldc2) {
  const int begin = (int)ptr[0];
  const int end = (int)ptr[1];
  for (int pos = blockIdx.x * blockDim.x + begin + threadIdx.x; pos < end;
      pos += blockDim.x * gridDim.x) {
    const Dtype valA = A[pos] * alpha;
    const int offset_part = (int)(indices[pos]) * ldc1;
    for (int colC = blockIdx.y * blockDim.y + threadIdx.y; colC < N;
        colC += blockDim.y * gridDim.y) {
      const int C_offset = offset_part + colC * ldc2;
      C[C_offset] = C[C_offset] + B[colC * ldb] * valA;
    }
  }
}

// C = alpha A * B^T +  C where A and B are vectors.
// A is a sprase vector and B is a dense vector
template<typename Dtype>
__device__ void tracker_gpu_csr_rank1_update_kernel(const int M, const int N,
                                                  const Dtype alpha,
                                                  const Dtype* A,
                                                  const Dtype* indices,
                                                  const Dtype* ptr,
                                                  const Dtype* B, int ldb,
                                                  Dtype* C,
                                                  const CBLAS_ORDER orderC) {
  if (orderC == CblasRowMajor) {
    tracker_gpu_csr_rank1_update_kernel_core(M, N, alpha, A, indices, ptr, B, ldb,
                                           C, N, 1);
  } else {
    tracker_gpu_csr_rank1_update_kernel_core(M, N, alpha, A, indices, ptr, B, ldb,
                                           C, 1, M);
  }
}

template<typename Dtype>
__global__ void tracker_gpu_csr_rank1_update_kernel_multi(
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A, const Dtype* indices, const Dtype* ptr,
    const Dtype* B, int ldb, Dtype* C, const CBLAS_ORDER orderC) {
  if (TransB == CblasNoTrans) {
    for (int i = 0; i < K; i++) {
      tracker_gpu_csr_rank1_update_kernel(M, N, alpha, A, indices, ptr + i,
                                        B + (N * i), 1, C, orderC);
    }
  } else {
    for (int i = 0; i < K; i++) {
      tracker_gpu_csr_rank1_update_kernel(M, N, alpha, A, indices, ptr + i, B + i,
                                        K, C, orderC);
    }
  }
}

template<>
void tracker_gpu_csr_gemm<float>(const CBLAS_TRANSPOSE TransA,
                               const CBLAS_TRANSPOSE TransB, const int M,
                               const int N, const int K, const float alpha,
                               int nzz, const float* A, const float* indices,
                               const float* ptr, const float* B, const float beta,
                               float* C, const CBLAS_ORDER orderC) {
  if (TransA == CblasNoTrans) {
    dim3 grids(M, N);
    dim3 threads(THREADS_PER_BLOCK_CSR, 1);
    tracker_gpu_csr_gemm_kernel<float><< <grids, threads>>>(TransB, M, N, K,
        alpha, nzz, A, indices, ptr, B, beta, C, orderC);
  } else {
    // scale C by beta
    if (beta != 1.0) {
      CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle() , M * N, &beta, C, 1));
    }
    const int average_nzz_per_row = nzz/K+1;
    dim3 grids((average_nzz_per_row+64-1)/64, N);
    dim3 threads(64, 1);
    tracker_gpu_csr_rank1_update_kernel_multi<float><< <grids, threads>>>(TransB,
        M, N, K,
        alpha, A, indices, ptr , B, 1, C, orderC);
  }
}

template<>
void tracker_gpu_csr_gemm<double>(const CBLAS_TRANSPOSE TransA,
                                const CBLAS_TRANSPOSE TransB, const int M,
                                const int N, const int K, const double alpha,
                                int nzz, const double* A, const double* indices,
                                const double* ptr, const double* B,
                                const double beta, double* C,
                                const CBLAS_ORDER orderC) {
  if (TransA == CblasNoTrans) {
    dim3 grids(M, N);
    dim3 threads(THREADS_PER_BLOCK_CSR, 1);
    tracker_gpu_csr_gemm_kernel<double><< <grids, threads>>> (TransB, M, N, K,
        alpha, nzz, A, indices, ptr, B, beta, C, orderC);
  } else {
    // scale C by beta
    if (beta != 1.0) {
      CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle() , M * N, &beta, C, 1));
    }
    const int average_nzz_per_row = nzz/K+1;
    dim3 grids((average_nzz_per_row+64-1)/64, N);
    dim3 threads(64, 1);
    tracker_gpu_csr_rank1_update_kernel_multi<double><< <grids, threads>>>(TransB,
        M, N, K,
        alpha, A, indices, ptr , B, 1, C, orderC);
  }
  CUDA_POST_KERNEL_CHECK;
}
}  // namespace caffe
