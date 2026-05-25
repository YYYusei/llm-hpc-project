
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

#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                                                        \
  do {                                                                          \
    cudaError_t _e = (call);                                                    \
    if (_e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,             \
              cudaGetErrorString(_e));                                          \
    }                                                                           \
  } while (0)
#endif

// Tiled GEMM kernel: C = alpha*A*B + beta*C
// Row-major storage.
// A: MxK, B: KxN, C: MxN
template <int BM, int BN, int BK>
__global__ void gemm_tiled_rm(const double* __restrict__ A,
                             const double* __restrict__ B,
                             double* __restrict__ C,
                             int M, int N, int K,
                             double alpha, double beta)
{
  // Block tile origin
  const int row0 = blockIdx.y * BM;
  const int col0 = blockIdx.x * BN;

  // Thread coordinates within the BMxBN tile
  const int tx = threadIdx.x; // [0, BN)
  const int ty = threadIdx.y; // [0, BM)

  // Shared memory tiles
  __shared__ double As[BM][BK];
  __shared__ double Bs[BK][BN];

  // Global indices this thread computes
  const int row = row0 + ty;
  const int col = col0 + tx;

  double acc = 0.0;

  // Loop over K dimension in chunks of BK
  for (int k0 = 0; k0 < K; k0 += BK) {
    // Load A tile: BMxBK
    if (row < M && (k0 + tx) < K && tx < BK) {
      As[ty][tx] = A[row * K + (k0 + tx)];
    } else if (tx < BK) {
      As[ty][tx] = 0.0;
    }

    // Load B tile: BKxBN
    if ((k0 + ty) < K && col < N && ty < BK) {
      Bs[ty][tx] = B[(k0 + ty) * N + col];
    } else if (ty < BK) {
      Bs[ty][tx] = 0.0;
    }

    __syncthreads();

    // Compute partial dot for this (row,col)
    if (row < M && col < N) {
#pragma unroll
      for (int kk = 0; kk < BK; ++kk) {
        acc += As[ty][kk] * Bs[kk][tx];
      }
    }

    __syncthreads();
  }

  // Write back
  if (row < M && col < N) {
    double c_old = C[row * N + col];
    C[row * N + col] = alpha * acc + beta * c_old;
  }
}

// Host launcher required by prompt.
// All pointers are device pointers, row-major.
void mm2_gpu(int ni, int nj, int nk, int nl,
             double alpha, double beta,
             double* A, double* B, double* C, double* D)
{
  // Allocate tmp on device: ni x nj
  double* tmp = nullptr;
  size_t tmp_bytes = (size_t)ni * (size_t)nj * sizeof(double);
  CHECK_CUDA(cudaMalloc((void**)&tmp, tmp_bytes));

  // Strategy: two tiled GEMMs
  // 1) tmp = alpha*A*B + 0*tmp
  // 2) D   = 1.0*tmp*C + beta*D

  // Tile sizes (simple, faithful tiled GEMM approach)
  constexpr int BM = 16;
  constexpr int BN = 16;
  constexpr int BK = 16;

  dim3 block(BN, BM, 1);

  // GEMM1: (ni x nk) * (nk x nj) -> (ni x nj)
  dim3 grid1((nj + BN - 1) / BN, (ni + BM - 1) / BM, 1);
  gemm_tiled_rm<BM, BN, BK><<<grid1, block>>>(A, B, tmp, ni, nj, nk, alpha, 0.0);

  // GEMM2: (ni x nj) * (nj x nl) -> (ni x nl), accumulate with beta*D
  dim3 grid2((nl + BN - 1) / BN, (ni + BM - 1) / BM, 1);
  gemm_tiled_rm<BM, BN, BK><<<grid2, block>>>(tmp, C, D, ni, nl, nj, 1.0, beta);

  // Cleanup
  CHECK_CUDA(cudaFree(tmp));
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
