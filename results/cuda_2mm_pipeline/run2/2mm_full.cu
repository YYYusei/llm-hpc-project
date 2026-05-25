
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

#ifndef TILE_M
#define TILE_M 16
#endif
#ifndef TILE_N
#define TILE_N 16
#endif
#ifndef TILE_K
#define TILE_K 16
#endif

#define CUDA_CHECK(call) do {                                   \
  cudaError_t _e = (call);                                      \
  if (_e != cudaSuccess) {                                      \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
            __FILE__, __LINE__, cudaGetErrorString(_e));        \
  }                                                             \
} while (0)

__global__ void gemm_tmp_kernel(int ni, int nj, int nk,
                                double alpha,
                                const double* __restrict__ A, // ni x nk
                                const double* __restrict__ B, // nk x nj
                                double* __restrict__ tmp)     // ni x nj
{
  __shared__ double As[TILE_M][TILE_K];
  __shared__ double Bs[TILE_K][TILE_N];

  int row = blockIdx.y * TILE_M + threadIdx.y;
  int col = blockIdx.x * TILE_N + threadIdx.x;

  double acc = 0.0;

  for (int k0 = 0; k0 < nk; k0 += TILE_K) {
    // Load A tile
    int a_col = k0 + threadIdx.x;
    if (row < ni && a_col < nk) As[threadIdx.y][threadIdx.x] = A[row * nk + a_col];
    else                       As[threadIdx.y][threadIdx.x] = 0.0;

    // Load B tile
    int b_row = k0 + threadIdx.y;
    if (b_row < nk && col < nj) Bs[threadIdx.y][threadIdx.x] = B[b_row * nj + col];
    else                        Bs[threadIdx.y][threadIdx.x] = 0.0;

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE_K; ++k) {
      acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < ni && col < nj) {
    tmp[row * nj + col] = alpha * acc;
  }
}

__global__ void gemm_D_kernel(int ni, int nj, int nl,
                              double beta,
                              const double* __restrict__ tmp, // ni x nj
                              const double* __restrict__ C,   // nj x nl
                              double* __restrict__ D)         // ni x nl (in/out)
{
  __shared__ double Ts[TILE_M][TILE_K];
  __shared__ double Cs[TILE_K][TILE_N];

  int row = blockIdx.y * TILE_M + threadIdx.y;
  int col = blockIdx.x * TILE_N + threadIdx.x;

  double acc = 0.0;

  for (int k0 = 0; k0 < nj; k0 += TILE_K) {
    // Load tmp tile
    int t_col = k0 + threadIdx.x;
    if (row < ni && t_col < nj) Ts[threadIdx.y][threadIdx.x] = tmp[row * nj + t_col];
    else                       Ts[threadIdx.y][threadIdx.x] = 0.0;

    // Load C tile
    int c_row = k0 + threadIdx.y;
    if (c_row < nj && col < nl) Cs[threadIdx.y][threadIdx.x] = C[c_row * nl + col];
    else                        Cs[threadIdx.y][threadIdx.x] = 0.0;

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE_K; ++k) {
      acc += Ts[threadIdx.y][k] * Cs[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < ni && col < nl) {
    double d0 = D[row * nl + col];
    D[row * nl + col] = acc + beta * d0;
  }
}

void mm2_gpu(int ni, int nj, int nk, int nl,
             double alpha, double beta,
             double* A, double* B, double* C, double* D)
{
  // Allocate tmp on device
  double* tmp = nullptr;
  size_t tmp_bytes = (size_t)ni * (size_t)nj * sizeof(double);
  CUDA_CHECK(cudaMalloc((void**)&tmp, tmp_bytes));
  if (!tmp) return;

  dim3 block(TILE_N, TILE_M);

  // Kernel 1: tmp = alpha * A * B
  dim3 grid1((nj + TILE_N - 1) / TILE_N,
             (ni + TILE_M - 1) / TILE_M);
  gemm_tmp_kernel<<<grid1, block>>>(ni, nj, nk, alpha, A, B, tmp);
  CUDA_CHECK(cudaGetLastError());

  // Kernel 2: D = tmp * C + beta * D
  dim3 grid2((nl + TILE_N - 1) / TILE_N,
             (ni + TILE_M - 1) / TILE_M);
  gemm_D_kernel<<<grid2, block>>>(ni, nj, nl, beta, tmp, C, D);
  CUDA_CHECK(cudaGetLastError());

  // Free tmp
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
