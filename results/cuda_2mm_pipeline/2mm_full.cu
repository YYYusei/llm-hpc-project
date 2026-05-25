
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

// ============ LLM Generated code (kernels + host launcher mm2_gpu) ============
#include <cuda_runtime.h>
#include <cstdio>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do {                                 \
  cudaError_t err__ = (call);                                  \
  if (err__ != cudaSuccess) {                                  \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
            __FILE__, __LINE__, cudaGetErrorString(err__));    \
  }                                                            \
} while (0)
#endif

// Fused blocked 2-stage GEMM:
// D = alpha*A*B*C + beta*D
// Row-major matrices:
// A: ni x nk, B: nk x nj, C: nj x nl, D: ni x nl
//
// Strategy implemented:
// - One threadblock computes one TM x TN tile of D.
// - Iterate k2 over nj in blocks of KB2.
//   For each k2-block:
//     1) Compute tmp_tile(TM x KB2) = sum_{k1} A(TM x K1) * B(K1 x KB2)
//     2) Accumulate Dacc(TM x TN) += alpha * tmp_tile(TM x KB2) * C(KB2 x TN)
// - Epilogue: D = beta*D + Dacc
//
// tmp_tile never stored to global memory.

template<int TM, int TN, int KB2, int K1>
__global__ void mm2_fused_kernel(int ni, int nj, int nk, int nl,
                                 double alpha, double beta,
                                 const double* __restrict__ A,
                                 const double* __restrict__ B,
                                 const double* __restrict__ C,
                                 double* __restrict__ D)
{
  // Block maps to one output tile D[i0:i0+TM, j0:j0+TN]
  const int i0 = blockIdx.y * TM;
  const int j0 = blockIdx.x * TN;

  // 2D threadblock: 16x16 = 256 threads
  const int tx = threadIdx.x; // [0,15]
  const int ty = threadIdx.y; // [0,15]
  const int tid = ty * blockDim.x + tx;

  // Each thread computes a 4x4 micro-tile of Dacc within the TMxTN tile.
  // With TM=TN=64 and blockDim=16x16, each thread covers:
  // rows: ty*4 + {0..3}, cols: tx*4 + {0..3}
  constexpr int RM = TM / 16; // 4
  constexpr int RN = TN / 16; // 4

  const int rBase = ty * RM;
  const int cBase = tx * RN;

  double Dacc[RM][RN];
#pragma unroll
  for (int rr = 0; rr < RM; ++rr)
#pragma unroll
    for (int cc = 0; cc < RN; ++cc)
      Dacc[rr][cc] = 0.0;

  extern __shared__ double shmem[];
  double* Asub = shmem;                              // TM*K1
  double* Bsub = Asub + TM * K1;                     // K1*KB2
  double* Csub = Bsub + K1 * KB2;                    // KB2*TN

  // Loop over k2 blocks (nj dimension) in chunks of KB2
  for (int k2 = 0; k2 < nj; k2 += KB2) {

    // Per-thread tmp fragment for current KB2 slice:
    // tmpFrag[rr][kb] corresponds to tmp_tile(i0+rBase+rr, k2+kb)
    double tmpFrag[RM][KB2];
#pragma unroll
    for (int rr = 0; rr < RM; ++rr)
#pragma unroll
      for (int kb = 0; kb < KB2; ++kb)
        tmpFrag[rr][kb] = 0.0;

    // Compute tmp_tile(TM x KB2) via k1 blocking over nk
    for (int k1 = 0; k1 < nk; k1 += K1) {

      // Cooperative load Asub (TM x K1)
      for (int idx = tid; idx < TM * K1; idx += blockDim.x * blockDim.y) {
        int r = idx / K1;
        int c = idx - r * K1;
        int gr = i0 + r;
        int gc = k1 + c;
        Asub[idx] = (gr < ni && gc < nk) ? A[gr * nk + gc] : 0.0;
      }

      // Cooperative load Bsub (K1 x KB2)
      for (int idx = tid; idx < K1 * KB2; idx += blockDim.x * blockDim.y) {
        int r = idx / KB2;
        int c = idx - r * KB2;
        int gr = k1 + r;
        int gc = k2 + c;
        Bsub[idx] = (gr < nk && gc < nj) ? B[gr * nj + gc] : 0.0;
      }

      __syncthreads();

      // Micro-kernel: accumulate tmpFrag += Asub(rows) * Bsub
#pragma unroll
      for (int kk = 0; kk < K1; ++kk) {
        double aReg[RM];
#pragma unroll
        for (int rr = 0; rr < RM; ++rr) {
          int r = rBase + rr;
          aReg[rr] = Asub[r * K1 + kk];
        }

#pragma unroll
        for (int kb = 0; kb < KB2; ++kb) {
          double b = Bsub[kk * KB2 + kb];
#pragma unroll
          for (int rr = 0; rr < RM; ++rr) {
            tmpFrag[rr][kb] += aReg[rr] * b;
          }
        }
      }

      __syncthreads();
    } // end k1 loop

    // Load Csub (KB2 x TN) for this k2 block
    for (int idx = tid; idx < KB2 * TN; idx += blockDim.x * blockDim.y) {
      int r = idx / TN;
      int c = idx - r * TN;
      int gr = k2 + r;
      int gc = j0 + c;
      Csub[idx] = (gr < nj && gc < nl) ? C[gr * nl + gc] : 0.0;
    }

    __syncthreads();

    // Accumulate Dacc += alpha * tmpFrag * Csub
#pragma unroll
    for (int kb = 0; kb < KB2; ++kb) {
      double cReg[RN];
#pragma unroll
      for (int cc = 0; cc < RN; ++cc) {
        int c = cBase + cc;
        cReg[cc] = Csub[kb * TN + c];
      }

#pragma unroll
      for (int rr = 0; rr < RM; ++rr) {
        double t = alpha * tmpFrag[rr][kb];
#pragma unroll
        for (int cc = 0; cc < RN; ++cc) {
          Dacc[rr][cc] += t * cReg[cc];
        }
      }
    }

    __syncthreads();
  } // end k2 loop

  // Epilogue: D = beta*D + Dacc
#pragma unroll
  for (int rr = 0; rr < RM; ++rr) {
    int gr = i0 + rBase + rr;
    if (gr >= ni) continue;
#pragma unroll
    for (int cc = 0; cc < RN; ++cc) {
      int gc = j0 + cBase + cc;
      if (gc >= nl) continue;
      double d0 = D[gr * nl + gc];
      D[gr * nl + gc] = beta * d0 + Dacc[rr][cc];
    }
  }
}

void mm2_gpu(int ni, int nj, int nk, int nl,
             double alpha, double beta,
             double* A, double* B, double* C, double* D)
{
  // Allocate tmp as required by the interface (not materialized by fused kernel).
  // This follows the requirement to allocate/free tmp inside mm2_gpu.
  double* tmp = nullptr;
  size_t tmpBytes = (size_t)ni * (size_t)nj * sizeof(double);
  CUDA_CHECK(cudaMalloc(&tmp, tmpBytes));

  // Launch fused kernel (tmp not used/stored).
  constexpr int TM  = 64;
  constexpr int TN  = 64;
  constexpr int KB2 = 16;
  constexpr int K1  = 16;

  dim3 block(16, 16, 1);
  dim3 grid((nl + TN - 1) / TN, (ni + TM - 1) / TM, 1);

  size_t shmemBytes = (TM * K1 + K1 * KB2 + KB2 * TN) * sizeof(double);

  mm2_fused_kernel<TM, TN, KB2, K1><<<grid, block, shmemBytes>>>(
      ni, nj, nk, nl, alpha, beta, A, B, C, D);

  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaFree(tmp));
}

// ============ Strict serial CPU reference ============
static void mm2_cpu(int ni,int nj,int nk,int nl,double alpha,double beta,
                    double* A,double* B,double* C,double* D,double* tmp) {
    for (int i=0;i<ni;i++)
        for (int j=0;j<nj;j++) {
            tmp[i*nj+j]=0.0;
            for (int k=0;k<nk;k++) tmp[i*nj+j]+=alpha*A[i*nk+k]*B[k*nj+j];
        }
    for (int i=0;i<ni;i++)
        for (int j=0;j<nl;j++) {
            D[i*nl+j]*=beta;
            for (int k=0;k<nj;k++) D[i*nl+j]+=tmp[i*nj+k]*C[k*nl+j];
        }
}

static void init_mats(int ni,int nj,int nk,int nl,
                      double* A,double* B,double* C,double* D) {
    for (int i=0;i<ni;i++) for (int k=0;k<nk;k++) A[i*nk+k]=(double)((i*k+1)%100)/ni;
    for (int k=0;k<nk;k++) for (int j=0;j<nj;j++) B[k*nj+j]=(double)((k+j)%100)/nj;
    for (int j=0;j<nj;j++) for (int l=0;l<nl;l++) C[j*nl+l]=(double)((j*l+2)%100)/nl;
    for (int i=0;i<ni;i++) for (int l=0;l<nl;l++) D[i*nl+l]=(double)((i+l)%100)/nl;
}

static double max_abs_diff(const double* a,const double* b,int n) {
    double m=0.0; for (int k=0;k<n;k++){double e=fabs(a[k]-b[k]); if(e>m)m=e;} return m;
}

int main() {
    int ni=NI,nj=NJ,nk=NK,nl=NL;
    double alpha=1.5, beta=1.2;

    double* A=(double*)malloc((size_t)ni*nk*sizeof(double));
    double* B=(double*)malloc((size_t)nk*nj*sizeof(double));
    double* C=(double*)malloc((size_t)nj*nl*sizeof(double));
    double* D0=(double*)malloc((size_t)ni*nl*sizeof(double));   // initial D
    double* Dc=(double*)malloc((size_t)ni*nl*sizeof(double));   // CPU result
    double* Dg=(double*)malloc((size_t)ni*nl*sizeof(double));   // GPU result
    double* tmp=(double*)malloc((size_t)ni*nj*sizeof(double));

    init_mats(ni,nj,nk,nl,A,B,C,D0);

    // ---- CPU reference (timed, 1 run; matmul is expensive) ----
    for (size_t k=0;k<(size_t)ni*nl;k++) Dc[k]=D0[k];
    clock_t cs=clock();
    mm2_cpu(ni,nj,nk,nl,alpha,beta,A,B,C,Dc,tmp);
    clock_t ce=clock();
    double cpu_ms=(double)(ce-cs)/CLOCKS_PER_SEC*1000.0;

    // ---- GPU ----
    double *dA,*dB,*dC,*dD;
    CHECK_CUDA(cudaMalloc(&dA,(size_t)ni*nk*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dB,(size_t)nk*nj*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dC,(size_t)nj*nl*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dD,(size_t)ni*nl*sizeof(double)));
    CHECK_CUDA(cudaMemcpy(dA,A,(size_t)ni*nk*sizeof(double),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB,B,(size_t)nk*nj*sizeof(double),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC,C,(size_t)nj*nl*sizeof(double),cudaMemcpyHostToDevice));

    // warmup + correctness
    CHECK_CUDA(cudaMemcpy(dD,D0,(size_t)ni*nl*sizeof(double),cudaMemcpyHostToDevice));
    mm2_gpu(ni,nj,nk,nl,alpha,beta,dA,dB,dC,dD);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(Dg,dD,(size_t)ni*nl*sizeof(double),cudaMemcpyDeviceToHost));

    // timed (5 runs)
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
