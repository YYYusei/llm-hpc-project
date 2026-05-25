
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { cudaError_t e=call; if(e!=cudaSuccess){printf("CUDA_ERROR: %s\n",cudaGetErrorString(e));exit(1);} }

#define N 2000
#define TSTEPS 20

// ============ LLM Generated code (kernels + host launcher jacobi_gpu) ============
#include <cuda_runtime.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do {                                  \
  cudaError_t err__ = (call);                                   \
  if (err__ != cudaSuccess) {                                   \
    /* In production you might handle/log this differently. */   \
  }                                                             \
} while (0)
#endif

__global__ void jacobi_stencil_2d(const double* __restrict__ in,
                                  double* __restrict__ out,
                                  int n)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x; // col
  int i = blockIdx.y * blockDim.y + threadIdx.y; // row

  if (i >= 1 && i <= n - 2 && j >= 1 && j <= n - 2) {
    int idx = i * n + j;
    out[idx] = 0.2 * (in[idx] +
                      in[idx - 1] +
                      in[idx + 1] +
                      in[idx + n] +
                      in[idx - n]);
  }
}

void jacobi_gpu(int tsteps, int n, double* A, double* B)
{
  // A and B are device pointers to row-major n*n arrays.
  if (tsteps <= 0 || n <= 2) return;

  dim3 block(32, 8);
  dim3 grid((n + block.x - 1) / block.x,
            (n + block.y - 1) / block.y);

  for (int t = 0; t < tsteps; ++t) {
    jacobi_stencil_2d<<<grid, block>>>(A, B, n); // B = stencil(A)
    jacobi_stencil_2d<<<grid, block>>>(B, A, n); // A = stencil(B)
  }

  // No host/device copies; caller owns synchronization policy.
  // If you want jacobi_gpu to be synchronous, uncomment:
  // CUDA_CHECK(cudaDeviceSynchronize());
}

// ============ Strict serial CPU reference ============
static void jacobi_cpu(int tsteps, int n, double* A, double* B) {
    for (int t = 0; t < tsteps; t++) {
        for (int i = 1; i < n - 1; i++)
            for (int j = 1; j < n - 1; j++)
                B[i*n+j] = 0.2 * (A[i*n+j] + A[i*n+(j-1)] + A[i*n+(1+j)] + A[(1+i)*n+j] + A[(i-1)*n+j]);
        for (int i = 1; i < n - 1; i++)
            for (int j = 1; j < n - 1; j++)
                A[i*n+j] = 0.2 * (B[i*n+j] + B[i*n+(j-1)] + B[i*n+(1+j)] + B[(1+i)*n+j] + B[(i-1)*n+j]);
    }
}

static void init_arrays(int n, double* A, double* B) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            A[i*n+j] = ((double)i*(j+2) + 2) / n;
            B[i*n+j] = ((double)i*(j+3) + 3) / n;
        }
}

static double max_abs_diff(const double* a, const double* b, int n) {
    double m = 0.0;
    for (int k = 0; k < n*n; k++) { double e = fabs(a[k]-b[k]); if (e>m) m=e; }
    return m;
}

int main() {
    int n = N, tsteps = TSTEPS;
    size_t bytes = (size_t)n*n*sizeof(double);

    double* A0 = (double*)malloc(bytes);
    double* B0 = (double*)malloc(bytes);
    double* Ac = (double*)malloc(bytes);
    double* Bc = (double*)malloc(bytes);
    double* Ag = (double*)malloc(bytes);

    init_arrays(n, A0, B0);

    // CPU reference (timed, 3 runs averaged)
    double cpu_ms = 0.0;
    for (int rep = 0; rep < 3; rep++) {
        for (size_t k = 0; k < (size_t)n*n; k++) { Ac[k]=A0[k]; Bc[k]=B0[k]; }
        clock_t s = clock();
        jacobi_cpu(tsteps, n, Ac, Bc);
        clock_t e = clock();
        cpu_ms += (double)(e-s)/CLOCKS_PER_SEC*1000.0;
    }
    cpu_ms /= 3.0;

    // GPU
    double *dA, *dB;
    CHECK_CUDA(cudaMalloc(&dA, bytes));
    CHECK_CUDA(cudaMalloc(&dB, bytes));

    // warmup + correctness
    CHECK_CUDA(cudaMemcpy(dA, A0, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, B0, bytes, cudaMemcpyHostToDevice));
    jacobi_gpu(tsteps, n, dA, dB);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(Ag, dA, bytes, cudaMemcpyDeviceToHost));

    // timed (5 runs)
    cudaEvent_t st, sp; cudaEventCreate(&st); cudaEventCreate(&sp);
    cudaEventRecord(st);
    for (int rep = 0; rep < 5; rep++) {
        CHECK_CUDA(cudaMemcpy(dA, A0, bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dB, B0, bytes, cudaMemcpyHostToDevice));
        jacobi_gpu(tsteps, n, dA, dB);
    }
    cudaEventRecord(sp); cudaEventSynchronize(sp);
    float gpu_ms = 0.0f; cudaEventElapsedTime(&gpu_ms, st, sp); gpu_ms /= 5.0f;

    double err = max_abs_diff(Ac, Ag, n);

    printf("BENCHMARK_RESULT:cpu_ms=%.4f,gpu_ms=%.4f,speedup=%.2f,error=%.2e\n",
           cpu_ms, gpu_ms, cpu_ms/(double)gpu_ms, err);

    cudaFree(dA); cudaFree(dB);
    free(A0); free(B0); free(Ac); free(Bc); free(Ag);
    return 0;
}
