#include <stdint.h>  // for uint32_t & uint64_t
#include <time.h>
#include <cmath>  // for std::fabs
#include <vector>
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/tracker_math.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

  template<typename Dtype>
class CsrFunctionsGenTest : public ::testing::Test {
 protected:
  CsrFunctionsGenTest()
      : A_(),
        indices_(),
        ptr_(),
        B_(),
        C_(),
        M(0),
        N(0),
        K(0),
        NZZ(0),
        PTR_SIZE(0),
        TransA(CblasNoTrans),
        TransB(CblasNoTrans),
        alpha(1.0),
        beta(0.0),
        orderC(CblasRowMajor) {
  }

  virtual void SetUp(int m, int n, int k, int nzz, int ptr_size) {
    M = m;
    N = n;
    K = k;
    NZZ = nzz;
    PTR_SIZE = ptr_size;

    A_.reset(new SyncedMemory(nzz * sizeof(Dtype)));
    indices_.reset(new SyncedMemory(nzz * sizeof(int)));
    ptr_.reset(new SyncedMemory(ptr_size * sizeof(int)));
    B_.reset(new SyncedMemory(K * N * sizeof(Dtype)));
    C_.reset(new SyncedMemory(M * N * sizeof(Dtype)));
  }

  virtual void run(bool isCpu, int times = 1) {
    if (isCpu) {
      Timer timer;
      timer.Start();
      for (int t = 0; t < times; t++) {
        tracker_cpu_csr_gemm(TransA, TransB, M, N, K, alpha, NZZ, cpu_A(),
                           cpu_indices(), cpu_ptr(), cpu_B(), beta, cpu_C(),
                           orderC);
      }
      std::cout << "Total Time for CSR CPU gemm M:" << M << " N: " << N
          << " K: " << K << " transA: " << TransA << " transB: " << TransB
          << " orderC: " << orderC << " equal to "
          << (timer.MilliSeconds() / times)
          << " milli seconds.. Time per M ops: "
          << timer.MilliSeconds() / (times * NZZ * N / 1e6)
          << " milli seconds\n";
    } else {
#ifndef CPU_ONLY
      Dtype* agpu = gpu_A();
      int* indicesgpu = gpu_indices();
      int* ptrgpu = gpu_ptr();
      Dtype* bgpu = gpu_B();
      Dtype* cgpu = gpu_C();
      Timer timer;
      timer.Start();
      for (int t = 0; t < times; t++) {
        tracker_gpu_csr_gemm_cusparse(TransA, TransB, M, N, K, alpha, NZZ, agpu,
                           indicesgpu, ptrgpu, bgpu, beta, cgpu, orderC);
      }
      cudaDeviceSynchronize();
      std::cout << "Total Time for CSR GPU gemm M:" << M << " N: " << N
          << " K: " << K << " transA: " << TransA << " transB: " << TransB
          << " orderC: " << orderC << " equal to "
          << (timer.MilliSeconds() / times)
          << " milli seconds. Time per M ops: "
          << timer.MilliSeconds() / (times * NZZ * N / 1e6)
          << " milli seconds\n";
#else

#endif
    }
  }

  void setA(Dtype A_data[], int A_indices[], int A_ptr[]) {
    Dtype* am = cpu_A();
    int* aindices = cpu_indices();
    int* aptr = cpu_ptr();

    for (int i = 0; i < NZZ; i++) {
      am[i] = A_data[i];
      aindices[i] = A_indices[i];
    }
    for (int i = 0; i < PTR_SIZE; i++) {
      aptr[i] = A_ptr[i];
    }
  }

  void setB(Dtype B_data[]) {
    Dtype* bm = cpu_B();
    for (int i = 0; i < (K * N); i++) {
      bm[i] = B_data[i];
    }
  }
  void setC(Dtype C_data[]) {
    Dtype* cm = cpu_C();
    for (int i = 0; i < (M * N); i++) {
      cm[i] = C_data[i];
    }
  }
  void checkC(Dtype C_check[]) {
    Dtype* cm = cpu_C();
    for (int i = 0; i < (M * N); i++) {
      EXPECT_EQ(cm[i], C_check[i]);
    }
  }

  Dtype* cpu_A() {
    CHECK(A_);
    return reinterpret_cast<Dtype*>(A_->mutable_cpu_data());
  }
  Dtype* gpu_A() {
    CHECK(A_);
    return reinterpret_cast<Dtype*>(A_->mutable_gpu_data());
  }

  Dtype* cpu_B() {
    CHECK(B_);
    return reinterpret_cast<Dtype*>(B_->mutable_cpu_data());
  }
  Dtype* gpu_B() {
    CHECK(B_);
    return reinterpret_cast<Dtype*>(B_->mutable_gpu_data());
  }

  Dtype* cpu_C() {
    CHECK(C_);
    return reinterpret_cast<Dtype*>(C_->mutable_cpu_data());
  }
  Dtype* gpu_C() {
    CHECK(C_);
    return reinterpret_cast<Dtype*>(C_->mutable_gpu_data());
  }

  int* cpu_indices() {
    CHECK(indices_);
    return reinterpret_cast<int*>(indices_->mutable_cpu_data());
  }
  int* gpu_indices() {
    CHECK(indices_);
    return reinterpret_cast<int*>(indices_->mutable_gpu_data());
  }

  int* cpu_ptr() {
    CHECK(ptr_);
    return reinterpret_cast<int*>(ptr_->mutable_cpu_data());
  }
  int* gpu_ptr() {
    CHECK(ptr_);
    return reinterpret_cast<int*>(ptr_->mutable_gpu_data());
  }

  void random_csr(int M, int N, int nzz_per_row, Dtype* A, int* indices,
                  int* ptr) {
    srand(0);
    ptr[0] = 0;
    for (int row = 0; row < M; row++) {
      ptr[row+1] = nzz_per_row * (row+1);
      for (int pos = 0; pos < nzz_per_row; pos++) {
        int col = caffe_rng_rand() % N;
        indices[row * nzz_per_row + pos] = col;
        A[row * nzz_per_row + pos] =
            static_cast <Dtype> (caffe_rng_rand()) /
            static_cast <Dtype> (RAND_MAX);
      }
    }
  }

  void random_fill(int size, Dtype* X) {
    srand(0);
    for (int pos = 0; pos < size; pos++) {
      X[pos] = static_cast<Dtype>(caffe_rng_rand()) /
          static_cast<Dtype>(RAND_MAX);
    }
  }

  void test_speed_forward(int batch_size, int features, int nzz_per_row,
                          int classes) {
    Dtype* A = new Dtype[batch_size * nzz_per_row];
    int* indices = new int[batch_size * nzz_per_row];
    int* ptr = new int[batch_size + 1];
    Dtype* B = new Dtype[features * classes];
    Dtype* C = new Dtype[batch_size * classes];
    this->random_csr(batch_size, features, nzz_per_row, A, indices, ptr);
    this->random_fill(features * classes, B);
    this->random_fill(batch_size * classes, C);

    this->alpha = 1.0;
    this->beta = 1.0;
    this->SetUp(batch_size, classes, features, batch_size * nzz_per_row,
                batch_size + 1);
    this->TransA = CblasNoTrans;
    this->TransB = CblasTrans;
    this->orderC = CblasRowMajor;

    this->setA(A, indices, ptr);
    this->setB(B);
    this->setC(C);
    this->run(true, 100);

    this->setC(C);
#ifndef CPU_ONLY
    this->run(false, 100);
#else
#endif
    delete A;
    delete indices;
    delete ptr;
    delete B;
    delete C;
  }

  void test_speed_backward(int batch_size, int features, int nzz_per_row,
                           int classes) {
    Dtype* A = new Dtype[batch_size * nzz_per_row];
    int* indices = new int[batch_size * nzz_per_row];
    int* ptr = new int[batch_size + 1];
    Dtype* B = new Dtype[batch_size * classes];
    Dtype* C = new Dtype[features * classes];
    this->random_csr(batch_size, features, nzz_per_row, A, indices, ptr);
    this->random_fill(batch_size * classes, B);
    this->random_fill(features * classes, C);

    this->alpha = 1.0;
    this->beta = 1.0;
    this->SetUp(features, classes, batch_size, batch_size * nzz_per_row,
                batch_size + 1);
    this->TransA = CblasTrans;
    this->TransB = CblasNoTrans;
    this->orderC = CblasColMajor;

    this->setA(A, indices, ptr);
    this->setB(B);
    this->setC(C);
    this->run(true, 100);

    this->setC(C);
#ifndef CPU_ONLY
    this->run(false, 100);
#else
#endif

    delete A;
    delete indices;
    delete ptr;
    delete B;
    delete C;
  }

  shared_ptr<SyncedMemory> A_;
  shared_ptr<SyncedMemory> indices_;
  shared_ptr<SyncedMemory> ptr_;
  shared_ptr<SyncedMemory> B_;
  shared_ptr<SyncedMemory> C_;
  int M;
  int N;
  int K;
  int NZZ;
  int PTR_SIZE;

  CBLAS_TRANSPOSE TransA;
  CBLAS_TRANSPOSE TransB;
  Dtype alpha;
  Dtype beta;
  CBLAS_ORDER orderC;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(CsrFunctionsGenTest, Dtypes);

TYPED_TEST(CsrFunctionsGenTest, TestCsrGemm1) {
TypeParam A[] = {1.0, 2.0, 3.0};
int indices[] = {0, 2, 1};
int ptr[] = {0, 2, 3};
TypeParam B[] = {4.0, 7.0, 5.0, 8.0, 6.0, 9.0};
TypeParam C[] = {0.0, 0.0, 0.0, 0.0};
TypeParam CCheck[] = {16.0, 25.0, 15.0, 24.0};
this->alpha = 1.0;
this->beta = 1.0;
this->SetUp(2, 2, 3, 3, 3);

this->setA(A, indices, ptr);
this->setB(B);
this->setC(C);
this->run(true);
this->checkC(CCheck);
this->setC(C);

#ifndef CPU_ONLY
this->run(false);
this->checkC(CCheck);
#else

#endif
}

TYPED_TEST(CsrFunctionsGenTest, TestCsrGemm2) {
TypeParam A[] = {1.0, 2.0, 3.0};
int indices[] = {0, 2, 1};
int ptr[] = {0, 2, 3};
TypeParam B[] = {4.0, 7.0, 5.0, 8.0, 6.0, 9.0};
TypeParam C[] = {1.0, 2.0, 3.0, 4.0};
TypeParam CCheck[] = {17.0, 27.0, 18.0, 28.0};
this->alpha = 1.0;
this->beta = 1.0;
this->SetUp(2, 2, 3, 3, 3);
this->TransA = CblasNoTrans;
this->TransB = CblasNoTrans;
this->orderC = CblasRowMajor;

this->setA(A, indices, ptr);
this->setB(B);
this->setC(C);
this->run(true);
this->checkC(CCheck);
this->setC(C);
#ifndef CPU_ONLY
this->run(false);
this->checkC(CCheck);
#else
#endif
}

TYPED_TEST(CsrFunctionsGenTest, TestCsrGemm3) {
TypeParam A[] = {1.0, 2.0, 3.0};
int indices[] = {0, 2, 1};
int ptr[] = {0, 2, 3};
TypeParam B[] = {4.0, 7.0, 5.0, 8.0, 6.0, 9.0};
TypeParam C[] = {1.0, 2.3, 3.0, 4.0};
TypeParam CCheck[] = {16.0, 25.0, 15.0, 24.0};
this->alpha = 1.0;
this->beta = 0.0;
this->SetUp(2, 2, 3, 3, 3);

this->setA(A, indices, ptr);
this->setB(B);
this->setC(C);
this->run(true);
this->checkC(CCheck);
this->setC(C);
#ifndef CPU_ONLY
this->run(false);
this->checkC(CCheck);
#else

#endif
}

TYPED_TEST(CsrFunctionsGenTest, TestCsrGemm4) {
TypeParam A[] = {1.0, 2.0, 3.0};
int indices[] = {0, 2, 1};
int ptr[] = {0, 2, 3};
TypeParam B[] = {4.0, 7.0, 5.0, 8.0, 6.0, 9.0};
TypeParam C[] = {0.0, 0.0, 0.0, 0.0};
TypeParam CCheck[] = {16.0, 15.0, 25.0, 24.0};
this->alpha = 1.0;
this->beta = 1.0;
this->SetUp(2, 2, 3, 3, 3);
this->TransA = CblasNoTrans;
this->TransB = CblasNoTrans;
this->orderC = CblasColMajor;

this->setA(A, indices, ptr);
this->setB(B);
this->setC(C);
this->run(true);
this->checkC(CCheck);
this->setC(C);
#ifndef CPU_ONLY
this->run(false);
this->checkC(CCheck);
#else

#endif
}

TYPED_TEST(CsrFunctionsGenTest, TestCsrGemm5) {
TypeParam A[] = {1.0, 2.0, 3.0};
int indices[] = {0, 2, 1};
int ptr[] = {0, 2, 3};
TypeParam B[] = {4.0, 7.0, 5.0, 8.0, 6.0, 9.0};
TypeParam C[] = {1.0, 2.0, 0.0, 0.0};
TypeParam CCheck[] = {17.0, 17.0, 25.0, 24.0};
this->alpha = 1.0;
this->beta = 1.0;
this->SetUp(2, 2, 3, 3, 3);
this->TransA = CblasNoTrans;
this->TransB = CblasNoTrans;
this->orderC = CblasColMajor;

this->setA(A, indices, ptr);
this->setB(B);
this->setC(C);
this->run(true);
this->checkC(CCheck);
this->setC(C);
#ifndef CPU_ONLY
this->run(false);
this->checkC(CCheck);
#else

#endif
}

TYPED_TEST(CsrFunctionsGenTest, TestCsrGemm6) {
TypeParam A[] = {1.0, 2.0, 3.0};
int indices[] = {0, 2, 1};
int ptr[] = {0, 2, 3};
TypeParam B[] = {4.0, 7.0, 5.0, 8.0, 6.0, 9.0};
TypeParam C[] = {1.0, 2.0, 3.0, 0.0};
TypeParam CCheck[] = {16.0, 15.0, 25.0, 24.0};
this->alpha = 1.0;
this->beta = 0.0;
this->SetUp(2, 2, 3, 3, 3);
this->TransA = CblasNoTrans;
this->TransB = CblasNoTrans;
this->orderC = CblasColMajor;

this->setA(A, indices, ptr);
this->setB(B);
this->setC(C);
this->run(true);
this->checkC(CCheck);
this->setC(C);
#ifndef CPU_ONLY
this->run(false);
this->checkC(CCheck);
#else

#endif
}

TYPED_TEST(CsrFunctionsGenTest, TestCsrGemm7) {
TypeParam A[] = {1.0, 2.0, 3.0};
int indices[] = {0, 2, 1};
int ptr[] = {0, 2, 3};
TypeParam B[] = {4.0, 7.0, 5.0, 8.0, 6.0, 9.0};
TypeParam C[] = {0.0, 0.0, 0.0, 0.0};
TypeParam CCheck[] = {32.0, 50.0, 30.0, 48.0};
this->alpha = 2.0;
this->beta = 1.0;
this->SetUp(2, 2, 3, 3, 3);

this->setA(A, indices, ptr);
this->setB(B);
this->setC(C);
this->run(true);
this->checkC(CCheck);
this->setC(C);
#ifndef CPU_ONLY
this->run(false);
this->checkC(CCheck);
#else

#endif
}

TYPED_TEST(CsrFunctionsGenTest, TestCsrGemm8) {
TypeParam A[] = {1.0, 2.0, 3.0};
int indices[] = {0, 2, 1};
int ptr[] = {0, 2, 3};
TypeParam B[] = {4.0, 7.0, 5.0, 8.0, 6.0, 9.0};
TypeParam C[] = {1.0, 2.0, 3.0, 0.0};
TypeParam CCheck[] = {31.0, 58.0, 51.0, 36.0};
this->alpha = 2.0;
this->beta = 3.0;
this->SetUp(2, 2, 3, 3, 3);
this->TransA = CblasNoTrans;
this->TransB = CblasTrans;
this->orderC = CblasRowMajor;

this->setA(A, indices, ptr);
this->setB(B);
this->setC(C);
this->run(true);
this->checkC(CCheck);
this->setC(C);
#ifndef CPU_ONLY
this->run(false);
this->checkC(CCheck);
#else

#endif
}

TYPED_TEST(CsrFunctionsGenTest, TestCsrGemm9) {
TypeParam A[] = {1.0, 2.0, 3.0};
int indices[] = {0, 2, 1};
int ptr[] = {0, 2, 3};
TypeParam B[] = {4.0, 7.0, 5.0, 8.0, 6.0, 9.0};
TypeParam C[] = {1.0, 2.0, 3.0, 0.0};
TypeParam CCheck[] = {31.0, 48.0, 61.0, 36.0};
this->alpha = 2.0;
this->beta = 3.0;
this->SetUp(2, 2, 3, 3, 3);
this->TransA = CblasNoTrans;
this->TransB = CblasTrans;
this->orderC = CblasColMajor;

this->setA(A, indices, ptr);
this->setB(B);
this->setC(C);
this->run(true);
this->checkC(CCheck);
this->setC(C);
#ifndef CPU_ONLY
this->run(false);
this->checkC(CCheck);
#else

#endif
}

TYPED_TEST(CsrFunctionsGenTest, TestCsrGemm10) {
TypeParam A[] = {1.0, 2.0, 3.0};
int indices[] = {0, 2, 1};
int ptr[] = {0, 2, 3};
TypeParam B[] = {4.0, 7.0, 5.0, 8.0, 6.0, 9.0};
TypeParam C[] = {1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
TypeParam CCheck[] = {11.0, 20.0, 19.0, 48.0, 36.0, 54.0, 16.0, 28.0, 20.0};
this->alpha = 2.0;
this->beta = 3.0;
this->SetUp(3, 3, 2, 3, 3);
this->TransA = CblasTrans;
this->TransB = CblasNoTrans;
this->orderC = CblasRowMajor;

this->setA(A, indices, ptr);
this->setB(B);
this->setC(C);
this->run(true);
this->checkC(CCheck);
this->setC(C);
#ifndef CPU_ONLY
this->run(false);
this->checkC(CCheck);
#else

#endif
}

TYPED_TEST(CsrFunctionsGenTest, TestCsrGemm11) {
TypeParam A[] = {1.0, 2.0, 3.0};
int indices[] = {0, 2, 1};
int ptr[] = {0, 2, 3};
TypeParam B[] = {4.0, 7.0, 5.0, 8.0, 6.0, 9.0};
TypeParam C[] = {1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
TypeParam CCheck[] = {11.0, 54.0, 25.0, 14.0, 36.0, 28.0, 10.0, 54.0, 20.0};
this->alpha = 2.0;
this->beta = 3.0;
this->SetUp(3, 3, 2, 3, 3);
this->TransA = CblasTrans;
this->TransB = CblasNoTrans;
this->orderC = CblasColMajor;

this->setA(A, indices, ptr);
this->setB(B);
this->setC(C);
this->run(true);
this->checkC(CCheck);
this->setC(C);
#ifndef CPU_ONLY
this->run(false);
this->checkC(CCheck);
#else

#endif
}

TYPED_TEST(CsrFunctionsGenTest, TestCsrGemm12) {
TypeParam A[] = {1.0, 2.0, 3.0};
int indices[] = {0, 2, 1};
int ptr[] = {0, 2, 3};
TypeParam B[] = {4.0, 7.0, 5.0, 8.0, 6.0, 9.0};
TypeParam C[] = {1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
TypeParam CCheck[] = {11.0, 16.0, 21.0, 42.0, 48.0, 54.0, 16.0, 20.0, 24.0};
this->alpha = 2.0;
this->beta = 3.0;
this->SetUp(3, 3, 2, 3, 3);
this->TransA = CblasTrans;
this->TransB = CblasTrans;
this->orderC = CblasRowMajor;

this->setA(A, indices, ptr);
this->setB(B);
this->setC(C);
this->run(true);
this->checkC(CCheck);
this->setC(C);
#ifndef CPU_ONLY
this->run(false);
this->checkC(CCheck);
#else

#endif
}

TYPED_TEST(CsrFunctionsGenTest, TestCsrGemm13) {
TypeParam A[] = {1.0, 2.0, 3.0};
int indices[] = {0, 2, 1};
int ptr[] = {0, 2, 3};
TypeParam B[] = {4.0, 7.0, 5.0, 8.0, 6.0, 9.0};
TypeParam C[] = {1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
TypeParam CCheck[] = {11.0, 48.0, 25.0, 10.0, 48.0, 20.0, 12.0, 54.0, 24.0};
this->alpha = 2.0;
this->beta = 3.0;
this->SetUp(3, 3, 2, 3, 3);
this->TransA = CblasTrans;
this->TransB = CblasTrans;
this->orderC = CblasColMajor;

this->setA(A, indices, ptr);
this->setB(B);
this->setC(C);
this->run(true);
this->checkC(CCheck);
this->setC(C);
#ifndef CPU_ONLY
this->run(false);
this->checkC(CCheck);
#else

#endif
}

TYPED_TEST(CsrFunctionsGenTest, TestCsrGemmSpeedForward) {
std::vector<int> batch_size;
std::vector<int> features;
std::vector<int> nzz_per_row;
std::vector<int> classes;

batch_size.push_back(64);
batch_size.push_back(128);
features.push_back(10000);
nzz_per_row.push_back(200);
classes.push_back(2);
classes.push_back(10);
classes.push_back(100);

for (int ba = 0; ba < batch_size.size(); ba++) {
  for (int f = 0; f < features.size(); f++) {
    for (int nr = 0; nr < nzz_per_row.size(); nr++) {
      for (int c = 0; c < classes.size(); c++) {
        this->test_speed_forward(batch_size[ba],
                                 features[f], nzz_per_row[nr], classes[c]);
      }
    }
  }
}
}

TYPED_TEST(CsrFunctionsGenTest, TestCsrGemmSpeedBackward) {
std::vector<int> batch_size;
std::vector<int> features;
std::vector<int> nzz_per_row;
std::vector<int> classes;

batch_size.push_back(64);
batch_size.push_back(128);
features.push_back(10000);
nzz_per_row.push_back(200);
classes.push_back(2);
classes.push_back(10);
classes.push_back(100);

for (int ba = 0; ba < batch_size.size(); ba++) {
  for (int f = 0; f < features.size(); f++) {
    for (int nr = 0; nr < nzz_per_row.size(); nr++) {
      for (int c = 0; c < classes.size(); c++) {
        this->test_speed_backward(batch_size[ba],
                                  features[f], nzz_per_row[nr], classes[c]);
      }
    }
  }
}
}



}  // namespace caffe
