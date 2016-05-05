#include "caffe/util/tracker_math.hpp"
#include "caffe/util/math_functions.hpp"
#include <iomanip>
#include <fstream>
#include "caffe/syncedmem.hpp"

namespace caffe {
  
  
  /* Turn X into its LU form, sotre pivot matrix  */
  //template <>
  //void tracker_cpu_sgetrf<float>(const int n, float* X) {
  //  
  //  magma_int_t magma_sgetrf(n, n, X, n, ipiv, info);
    
  //}
  
  /* Solve linear equation A X = B given LU factorization of A */
  //template <>
  //void tracker_cpu_sgetrs<float>(const int n, float* X) {
      
  //}
  
  
  template <>
  void tracker_cpu_inverse<float>(const int n, float* X) {
    /*  
     *            Calculates the inverse of the n*n matrix X: Y = Y^-1
     *                See http://itf.fys.kuleuven.be/~rob/computer/lapack_wrapper/
     *            Does not change the value of X, unless Y=X
     */
    
    int info=0;
    
    
    /*  We need to store the pivot matrix obtained by the LU factorisation  */
    int *ipiv;
    ipiv= (int*) malloc(n*sizeof(int));
    if (ipiv==NULL) {
      LOG(FATAL) << "malloc failed in matrix_invert";
    }
    
    /*  Turn X into its LU form, store pivot matrix  */
    info = clapack_sgetrf (CblasRowMajor, n, n, X, n, ipiv);
    
    /*  Don't bother continuing when illegal argument (info<0) or singularity (info>0) occurs  */
    if (info!=0) LOG(FATAL) << "Matrix inversion failed: " << info;
    
    /*  Feed this to the lapack inversion routine.  */
    info = clapack_sgetri (CblasRowMajor, n, X, n, ipiv);
    
    /*  Cleanup */
    free(ipiv);
  }
  
  
  template <>
  void tracker_cpu_inverse<double>(const int n, double* X) {
    /*  
     *                See http://itf.fys.kuleuven.be/~rob/computer/lapack_wrapper/
     *            Calculates the inverse of the n*n matrix X: Y = X^-1
     *            Does not change the value of X, unless Y=X
     */
    
    int info=0;
    
    /*  We need to store the pivot matrix obtained by the LU factorisation  */
    int *ipiv;
    ipiv= (int*) malloc(n*sizeof(int));
    if (ipiv==NULL) {
      LOG(FATAL) << "malloc failed in matrix_invert";
    }
    
    /*  Turn X into its LU form, store pivot matrix  */
    info = clapack_dgetrf (CblasRowMajor, n, n, X, n, ipiv);
    
    /*  Don't bother continuing when illegal argument (info<0) or singularity (info>0) occurs  */
    if (info!=0) LOG(FATAL) << "Matrix inversion failed: " << info;
    
    /*  Feed this to the lapack inversion routine.  */
    info = clapack_dgetri (CblasRowMajor, n, X, n, ipiv);
    
    /*  Cleanup  */
    free(ipiv);
  }
  

template <>
void tracker_strided_add_scalar(const int N, const float alpha, int incx, float* Y) {
  for (int i = 0; i < N; i += incx) {
    Y[i] += alpha;
  }
}

template <>
void tracker_strided_add_scalar(const int N, const double alpha, int incx, double* Y) {
  for (int i = 0; i < N; i += incx) {
    Y[i] += alpha;
  }
}

template <typename Dtype>
void tracker_printMat(std::ostream& buffer, const Dtype* mat, int col, int count) {
  buffer << std::fixed << std::setprecision(5);
  for (int i = 0; i < count; ++i) {
    if(i != 0 && i % col == 0) {
      buffer << ';' << std::endl;
    }
    buffer << std::setw(10) << mat[i] << ' ';
  }
  buffer << std::endl;
}


template void tracker_printMat<double>(std::ostream& buffer, const double* mat, int col, int count);
template void tracker_printMat<float>(std::ostream& buffer, const float* mat, int col, int count);
template void tracker_printMat<int>(std::ostream& buffer, const int* mat, int col, int count);


template <typename Dtype>
void tracker_saveMat(string filename, const Dtype* mat, int col, int count) {
  std::ofstream outfile;
  outfile.open(filename.c_str(), std::ios::out | std::ios::binary);
  outfile.write((char*)&col, sizeof(int));
  outfile.write((char*)&count, sizeof(int));
  outfile.write((char*)mat, count*sizeof(Dtype));
  outfile.close();
}


template void tracker_saveMat<float>(string filename, const float* mat, int col, int count);
template void tracker_saveMat<double>(string filename, const double* mat, int col, int count);



template<typename Dtype>
void tracker_cpu_copy(const Dtype* A, const int ina, const int lda, 
                      Dtype* B, const int inb, const int ldb, 
                      const int nrows, const int ncols) {
   for(int j = 0; j < nrows; j++) {
      for(int i = 0; i < ncols; i++) {
         B[i * ina] = A[i * inb];
      }
      A += lda;
      B += ldb;
   }
}

template void tracker_cpu_copy<float>(const float* A, const int ina, const int lda, float* B, const int inb, const int ldb, const int nrows, const int ncols);
template void tracker_cpu_copy<double>(const double* A, const int ina, const int lda, double* B, const int inb, const int ldb, const int nrows, const int ncols);

template<>
void tracker_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M, 
                             const int N, const int K, const float alpha, const float* A, 
                             const int lda, const float* B, const int ldb, const float beta, 
                             float* C) {
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, N);
}
                           
template<>
void tracker_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M, 
                            const int N, const int K, const double alpha, const double* A, 
                            const int lda, const double* B,  const int ldb, const double beta, 
                            double* C) {
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, N);
}

template <typename Dtype>
void tracker_cpu_transpose(const int m, const int n, const Dtype* a, const int lda, Dtype* b, const int ldb) {
  if(a == b && (lda != n || ldb != m)) {
    LOG(FATAL) << "For inplace transpose data has to be contiguous in memory";
  }
  
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < n; j++) {
      const int from = i * lda + j;
      const int to = j * ldb + i;
      
      //LOG(ERROR) << from << ", " << to << ", " << m << ", " << n << ", " << lda << ", " << ldb;
      if(a == b && j > i) 
      {
        const Dtype tmp = b[from];
        b[from] = b[to];
        b[to] = tmp;
      } else {
        b[to] = a[from];
      }
    }
  }
}

template void tracker_cpu_transpose<float>(const int m, const int n, const float* a, const int lda, float* b, const int ldb);
template void tracker_cpu_transpose<double>(const int m, const int n, const double* a, const int lda, double* b, const int ldb);


template<typename Dtype>
void tracker_cpu_csr_gemm(const CBLAS_TRANSPOSE TransA,
                        const CBLAS_TRANSPOSE TransB, const int M, const int N,
                        const int K, const Dtype alpha, const int nzz,
                        const Dtype* A, const Dtype* indices, const Dtype* ptr,
                        const Dtype* B, const Dtype beta, Dtype* C,
                        const CBLAS_ORDER orderC) {
  if (TransA == CblasNoTrans) {  // CSR
    caffe_scal(M * N, beta, C);
    if (orderC == CblasRowMajor) {
      if (TransB == CblasNoTrans) {
        for (int rowA = 0; rowA < M; rowA++) {
          const int begin = (int)ptr[rowA];
          const int end = (int)ptr[rowA + 1];
          Dtype* CrowA = C + (N * rowA);
          for (int pos = begin; pos < end; pos++) {
            const Dtype* BcolAN = B + ((int)(indices[pos]) * N);
            const Dtype AatPos = alpha * A[pos];
            tracker_axpy(N, AatPos, BcolAN, CrowA, 1, 1);
          }
        }
      } else {
        for (int rowA = 0; rowA < M; rowA++) {
          const int begin = (int)ptr[rowA];
          const int end = (int)ptr[rowA + 1];
          Dtype* CrowA = C + (N * rowA);
          for (int pos = begin; pos < end; pos++) {
            const Dtype AatPos = alpha * A[pos];
            const Dtype* BcolA = B + (int)indices[pos];
            tracker_axpy(N, AatPos, BcolA, CrowA, K, 1);
          }
        }
      }
    } else {
      if (TransB == CblasNoTrans) {
        for (int rowA = 0; rowA < M; rowA++) {
          const int begin = (int)ptr[rowA];
          const int end = (int)ptr[rowA + 1];
          Dtype* CrowA = C + rowA;
          for (int pos = begin; pos < end; pos++) {
            const Dtype* BcolAN = B + ((int)(indices[pos]) * N);
            const Dtype AatPos = alpha * A[pos];
            tracker_axpy(N, AatPos, BcolAN, CrowA, 1, M);
          }
        }
      } else {
        for (int rowA = 0; rowA < M; rowA++) {
          const int begin = (int)ptr[rowA];
          const int end = (int)ptr[rowA + 1];
          Dtype* CrowA = C + rowA;
          for (int pos = begin; pos < end; pos++) {
            const Dtype* BcolA = B + (int)indices[pos];
            const Dtype AatPos = alpha * A[pos];
            tracker_axpy(N, AatPos, BcolA, CrowA, K, M);
          }
        }
      }
    }
  } else {  // A is CSC
    caffe_scal(M * N, beta, C);
    if (orderC == CblasRowMajor) {
      if (TransB == CblasNoTrans) {
        for (int colA = 0; colA < K; colA++) {
          const int begin = (int)ptr[colA];
          const int end = (int)ptr[colA + 1];
          const Dtype* BColAN = B + (colA * N);
          for (int pos = begin; pos < end; pos++) {
            tracker_axpy(N, A[pos] * alpha, BColAN,
                         C + ((int)(indices[pos]) * N), 1, 1);
          }
        }
      } else {
        for (int colA = 0; colA < K; colA++) {
          const int begin = (int)ptr[colA];
          const int end = (int)ptr[colA + 1];
          const Dtype* BColA = B + colA;
          for (int pos = begin; pos < end; pos++) {
            tracker_axpy(N, A[pos] * alpha, BColA, C + ((int)(indices[pos]) * N),
                       K, 1);
          }
        }
      }
    } else {
      if (TransB == CblasNoTrans) {
        for (int colA = 0; colA < K; colA++) {
          const int begin = (int)ptr[colA];
          const int end = (int)ptr[colA + 1];
          const Dtype* BColAN = B + (colA * N);
          for (int pos = begin; pos < end; pos++) {
            tracker_axpy(N, A[pos] * alpha, BColAN, C + (int)indices[pos], 1, M);
          }
        }
        
      } else {
        for (int colA = 0; colA < K; colA++) {
          const int begin = (int)ptr[colA];
          const int end = (int)ptr[colA + 1];
          const Dtype* BColA = B + colA;
          for (int pos = begin; pos < end; pos++) {
            tracker_axpy(N, A[pos] * alpha, BColA, C + (int)indices[pos], K,  M);
          }
        }
      }
    }
  }
}
                        
template void tracker_cpu_csr_gemm<float>(const CBLAS_TRANSPOSE TransA,
                                        const CBLAS_TRANSPOSE TransB,
                                        const int M, const int N, const int K,
                                        const float alpha, const int nzz,
                                        const float* A, const float* indices,
                                        const float* ptr, const float* B,
                                        const float beta, float* C,
                                        const CBLAS_ORDER orderC);
                        
template void tracker_cpu_csr_gemm<double>(const CBLAS_TRANSPOSE TransA,
                                         const CBLAS_TRANSPOSE TransB,
                                         const int M, const int N, const int K,
                                         const double alpha, const int nzz,
                                         const double* A, const double* indices,
                                         const double* ptr, const double* B,
                                         const double beta, double* C,
                                         const CBLAS_ORDER orderC);

template<>
void tracker_axpy<float>(const int N, const float alpha, const float* X, float* Y,
                       const int ldx, const int ldy) {
  cblas_saxpy(N, alpha, X, ldx, Y, ldy);
}
                       
template<>
void tracker_axpy<double>(const int N, const double alpha, const double* X,
                        double* Y, const int ldx, const int ldy) {
  cblas_daxpy(N, alpha, X, ldx, Y, ldy);
}
                        
}  // namespace caffe
