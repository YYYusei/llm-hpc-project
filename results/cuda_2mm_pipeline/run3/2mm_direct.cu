
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
#include <cstdlib>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(err__));                               \
      std::abort();                                                          \
    }                                                                        \
  } while (0)
#endif

// tmp(i,j) = alpha * sum_k A(i,k) * B(k,j)
// A: ni x nk, B: nk x nj, tmp: ni x nj
__global__ void k_tmp_ab(int ni, int nj, int nk,
                         double alpha,
                         const double* __restrict__ A,
                         const double* __restrict__ B,
                         double* __restrict__ tmp) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ni || j >= nj) return;

  double acc = 0.0;
  const int a_row = i * nk;
  for (int k = 0; k < nk; ++k) {
    acc += A[a_row + k] * B[k * nj + j];
  }
  tmp[i * nj + j] = alpha * acc;
}

// D(i,j) = sum_k tmp(i,k) * C(k,j) + beta * D(i,j)
// tmp: ni x nj, C: nj x nl, D: ni x nl
__global__ void k_d_tmpc(int ni, int nj, int nl,
                         double beta,
                         const double* __restrict__ tmp,
                         const double* __restrict__ C,
                         double* __restrict__ D) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ni || j >= nl) return;

  double acc = 0.0;
  const int tmp_row = i * nj;
  for (int k = 0; k < nj; ++k) {
    acc += tmp[tmp_row + k] * C[k * nl + j];
  }
  const int idx = i * nl + j;
  D[idx] = acc + beta * D[idx];
}

void mm2_gpu(int ni, int nj, int nk, int nl,
             double alpha, double beta,
             double* A, double* B, double* C, double* D) {
  // Allocate device scratch tmp (ni * nj)
  double* tmp = nullptr;
  size_t tmp_bytes = (size_t)ni * (size_t)nj * sizeof(double);
  CUDA_CHECK(cudaMalloc((void**)&tmp, tmp_bytes));

  // Launch 1: tmp = alpha * A * B
  dim3 block1(16, 16);
  dim3 grid1((nj + block1.x - 1) / block1.x,
             (ni + block1.y - 1) / block1.y);
  k_tmp_ab<<<grid1, block1>>>(ni, nj, nk, alpha, A, B, tmp);
  CUDA_CHECK(cudaGetLastError());

  // Launch 2: D = tmp * C + beta * D
  dim3 block2(16, 16);
  dim3 grid2((nl + block2.x - 1) / block2.x,
             (ni + block2.y - 1) / block2.y);
  k_d_tmpc<<<grid2, block2>>>(ni, nj, nl, beta, tmp, C, D);
  CUDA_CHECK(cudaGetLastError());

  // Ensure kernels complete before freeing tmp
  CUDA_CHECK(cudaDeviceSynchronize());
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
