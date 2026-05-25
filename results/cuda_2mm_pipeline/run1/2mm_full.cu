
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#define CHECK_CUDA(call) { cudaError_t e=call; if(e!=cudaSuccess){printf("CUDA_ERROR: %s\n",cudaGetErrorString(e));exit(1);} }
#define NI 1024
#define NJ 1024
#define NK 1024
#define NL 1024
// ===== LLM Generated (kernels + host launcher mm2_gpu) =====
#include <cuda_runtime.h>
#include <cstdio>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do {                                     \
  cudaError_t _e = (call);                                        \
  if (_e != cudaSuccess) {                                        \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                     \
            __FILE__, __LINE__, cudaGetErrorString(_e));          \
  }                                                               \
} while (0)
#endif

// Strategy-faithful fused blocked kernel: computes D tile directly without global tmp.
// Row-major: A[ni][nk], B[nk][nj], C[nj][nl], D[ni][nl].
template<int BM, int BN, int BK1, int BK2>
__global__ void mm2_fused_kernel(int ni, int nj, int nk, int nl,
                                 double alpha, double beta,
                                 const double* __restrict__ A,
                                 const double* __restrict__ B,
                                 const double* __restrict__ C,
                                 double* __restrict__ D)
{
  // Tile origin for this block in D
  const int i0 = blockIdx.y * BM;
  const int j0 = blockIdx.x * BN;

  // 2D threadblock mapping onto BMxBN tile
  // Use 16x16 threads (256 threads) by default with BM=BN=64 => each thread computes 4x4 outputs.
  const int tx = threadIdx.x; // [0..15]
  const int ty = threadIdx.y; // [0..15]

  constexpr int TM = BM / 16; // 4
  constexpr int TN = BN / 16; // 4

  const int iBase = i0 + ty * TM;
  const int jBase = j0 + tx * TN;

  // Shared memory tiles as specified by strategy
  __shared__ double A_sh[BM][BK1];   // (i x k1)
  __shared__ double B_sh[BK1][BK2];  // (k1 x k2)
  __shared__ double C_sh[BK2][BN];   // (k2 x j)

  // Accumulator for D subtile (TM x TN)
  double acc[TM][TN];
#pragma unroll
  for (int ii = 0; ii < TM; ++ii)
#pragma unroll
    for (int jj = 0; jj < TN; ++jj)
      acc[ii][jj] = 0.0;

  // Loop over k2 in chunks BK2 (streaming over nj)
  for (int k2 = 0; k2 < nj; k2 += BK2) {

    // Load C_sh[k2:k2+BK2, j0:j0+BN] (coalesced along j)
    // Each thread loads multiple elements to cover BK2*BN.
    for (int kk = ty; kk < BK2; kk += blockDim.y) {
      for (int jj = tx; jj < BN; jj += blockDim.x) {
        const int gk = k2 + kk;
        const int gj = j0 + jj;
        double v = 0.0;
        if (gk < nj && gj < nl) v = C[gk * nl + gj];
        C_sh[kk][jj] = v;
      }
    }
    __syncthreads();

    // tmp_reg fragment for this thread: TM x BK2 (small fragment of tmp for its i-subtile and k2-subtile)
    double tmp_reg[TM][BK2];
#pragma unroll
    for (int ii = 0; ii < TM; ++ii)
#pragma unroll
      for (int kk = 0; kk < BK2; ++kk)
        tmp_reg[ii][kk] = 0.0;

    // Compute tmp fragment for this k2-chunk by looping k1 in chunks BK1
    for (int k1 = 0; k1 < nk; k1 += BK1) {

      // Load A_sh[i0:i0+BM, k1:k1+BK1] (coalesced along k1)
      // Cover BM*BK1 elements.
      for (int ii = ty; ii < BM; ii += blockDim.y) {
        for (int kk = tx; kk < BK1; kk += blockDim.x) {
          const int gi = i0 + ii;
          const int gk = k1 + kk;
          double v = 0.0;
          if (gi < ni && gk < nk) v = A[gi * nk + gk];
          A_sh[ii][kk] = v;
        }
      }

      // Load B_sh[k1:k1+BK1, k2:k2+BK2] (coalesced along k2)
      // Cover BK1*BK2 elements.
      for (int kk1 = ty; kk1 < BK1; kk1 += blockDim.y) {
        for (int kk2 = tx; kk2 < BK2; kk2 += blockDim.x) {
          const int gk1 = k1 + kk1;
          const int gk2 = k2 + kk2;
          double v = 0.0;
          if (gk1 < nk && gk2 < nj) v = B[gk1 * nj + gk2];
          B_sh[kk1][kk2] = v;
        }
      }

      __syncthreads();

      // tmp_reg += alpha * (A_sh * B_sh) for this thread's TM rows and all BK2 cols
#pragma unroll
      for (int kk1 = 0; kk1 < BK1; ++kk1) {
        double a_frag[TM];
#pragma unroll
        for (int ii = 0; ii < TM; ++ii) {
          const int li = (ty * TM) + ii; // local row within BM
          a_frag[ii] = A_sh[li][kk1];
        }
#pragma unroll
        for (int kk2 = 0; kk2 < BK2; ++kk2) {
          const double b = B_sh[kk1][kk2];
#pragma unroll
          for (int ii = 0; ii < TM; ++ii) {
            tmp_reg[ii][kk2] += alpha * a_frag[ii] * b;
          }
        }
      }

      __syncthreads();
    } // k1

    // acc_reg += tmp_reg * C_sh  (BMxBK2)*(BK2xBN) -> BMxBN, per-thread TMxTN
#pragma unroll
    for (int kk2 = 0; kk2 < BK2; ++kk2) {
      double c_frag[TN];
#pragma unroll
      for (int jj = 0; jj < TN; ++jj) {
        const int lj = (tx * TN) + jj; // local col within BN
        c_frag[jj] = C_sh[kk2][lj];
      }
#pragma unroll
      for (int ii = 0; ii < TM; ++ii) {
        const double t = tmp_reg[ii][kk2];
#pragma unroll
        for (int jj = 0; jj < TN; ++jj) {
          acc[ii][jj] += t * c_frag[jj];
        }
      }
    }

    __syncthreads();
  } // k2

  // Write D = beta*D + acc (read D once, scale, add, store)
#pragma unroll
  for (int ii = 0; ii < TM; ++ii) {
    const int gi = iBase + ii;
    if (gi >= ni) continue;
#pragma unroll
    for (int jj = 0; jj < TN; ++jj) {
      const int gj = jBase + jj;
      if (gj >= nl) continue;
      const int idx = gi * nl + gj;
      const double d0 = D[idx];
      D[idx] = beta * d0 + acc[ii][jj];
    }
  }
}

// Host launcher required by prompt.
// NOTE: tmp is allocated/freed as device scratch to satisfy interface requirement,
// but the chosen strategy fuses and does not materialize tmp to global memory.
void mm2_gpu(int ni, int nj, int nk, int nl,
             double alpha, double beta,
             double* A, double* B, double* C, double* D)
{
  // Allocate tmp scratch as requested (unused by fused strategy).
  double* tmp = nullptr;
  size_t tmp_bytes = (size_t)ni * (size_t)nj * sizeof(double);
  CUDA_CHECK(cudaMalloc(&tmp, tmp_bytes));

  // Strategy tile sizes (recommended for FP64)
  constexpr int BM  = 64;
  constexpr int BN  = 64;
  constexpr int BK1 = 16;
  constexpr int BK2 = 16;

  dim3 block(16, 16, 1); // 256 threads, 2D threadblock
  dim3 grid((nl + BN - 1) / BN,
            (ni + BM - 1) / BM,
            1);

  mm2_fused_kernel<BM, BN, BK1, BK2><<<grid, block>>>(
      ni, nj, nk, nl, alpha, beta, A, B, C, D);

  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaFree(tmp));
}
// ===== Strict serial CPU reference =====
static void mm2_cpu(int ni,int nj,int nk,int nl,double alpha,double beta,
                    double* A,double* B,double* C,double* D,double* tmp) {
    for (int i=0;i<ni;i++) for (int j=0;j<nj;j++) {
        tmp[i*nj+j]=0.0;
        for (int k=0;k<nk;k++) tmp[i*nj+j]+=alpha*A[i*nk+k]*B[k*nj+j];
    }
    for (int i=0;i<ni;i++) for (int j=0;j<nl;j++) {
        D[i*nl+j]*=beta;
        for (int k=0;k<nj;k++) D[i*nl+j]+=tmp[i*nj+k]*C[k*nl+j];
    }
}
static void init_mats(int ni,int nj,int nk,int nl,double* A,double* B,double* C,double* D) {
    for (int i=0;i<ni;i++) for (int k=0;k<nk;k++) A[i*nk+k]=(double)((i*k+1)%100)/ni;
    for (int k=0;k<nk;k++) for (int j=0;j<nj;j++) B[k*nj+j]=(double)((k+j)%100)/nj;
    for (int j=0;j<nj;j++) for (int l=0;l<nl;l++) C[j*nl+l]=(double)((j*l+2)%100)/nl;
    for (int i=0;i<ni;i++) for (int l=0;l<nl;l++) D[i*nl+l]=(double)((i+l)%100)/nl;
}
static double max_abs_diff(const double* a,const double* b,int n){double m=0;for(int k=0;k<n;k++){double e=fabs(a[k]-b[k]);if(e>m)m=e;}return m;}
int main() {
    int ni=NI,nj=NJ,nk=NK,nl=NL; double alpha=1.5, beta=1.2;
    double* A=(double*)malloc((size_t)ni*nk*sizeof(double));
    double* B=(double*)malloc((size_t)nk*nj*sizeof(double));
    double* C=(double*)malloc((size_t)nj*nl*sizeof(double));
    double* D0=(double*)malloc((size_t)ni*nl*sizeof(double));
    double* Dc=(double*)malloc((size_t)ni*nl*sizeof(double));
    double* Dg=(double*)malloc((size_t)ni*nl*sizeof(double));
    double* tmp=(double*)malloc((size_t)ni*nj*sizeof(double));
    init_mats(ni,nj,nk,nl,A,B,C,D0);
    for (size_t k=0;k<(size_t)ni*nl;k++) Dc[k]=D0[k];
    clock_t cs=clock(); mm2_cpu(ni,nj,nk,nl,alpha,beta,A,B,C,Dc,tmp); clock_t ce=clock();
    double cpu_ms=(double)(ce-cs)/CLOCKS_PER_SEC*1000.0;
    double *dA,*dB,*dC,*dD;
    CHECK_CUDA(cudaMalloc(&dA,(size_t)ni*nk*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dB,(size_t)nk*nj*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dC,(size_t)nj*nl*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dD,(size_t)ni*nl*sizeof(double)));
    CHECK_CUDA(cudaMemcpy(dA,A,(size_t)ni*nk*sizeof(double),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB,B,(size_t)nk*nj*sizeof(double),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC,C,(size_t)nj*nl*sizeof(double),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dD,D0,(size_t)ni*nl*sizeof(double),cudaMemcpyHostToDevice));
    mm2_gpu(ni,nj,nk,nl,alpha,beta,dA,dB,dC,dD);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(Dg,dD,(size_t)ni*nl*sizeof(double),cudaMemcpyDeviceToHost));
    cudaEvent_t st,sp; cudaEventCreate(&st); cudaEventCreate(&sp);
    cudaEventRecord(st);
    for (int r=0;r<5;r++) {
        CHECK_CUDA(cudaMemcpy(dD,D0,(size_t)ni*nl*sizeof(double),cudaMemcpyHostToDevice));
        mm2_gpu(ni,nj,nk,nl,alpha,beta,dA,dB,dC,dD);
    }
    cudaEventRecord(sp); cudaEventSynchronize(sp);
    float gpu_ms=0.0f; cudaEventElapsedTime(&gpu_ms,st,sp); gpu_ms/=5.0f;
    double err=max_abs_diff(Dc,Dg,ni*nl);
    printf("BENCHMARK_RESULT:cpu_ms=%.4f,gpu_ms=%.4f,speedup=%.2f,error=%.2e\n",
           cpu_ms,gpu_ms,cpu_ms/(double)gpu_ms,err);
    cudaFree(dA);cudaFree(dB);cudaFree(dC);cudaFree(dD);
    free(A);free(B);free(C);free(D0);free(Dc);free(Dg);free(tmp);
    return 0;
}
