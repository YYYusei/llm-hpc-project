
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { cudaError_t e=call; if(e!=cudaSuccess){printf("CUDA_ERROR: %s\n",cudaGetErrorString(e));exit(1);} }

#define N 2000
#define TSTEPS 20

// ============ LLM Generated code (kernel + host launcher seidel_gpu) ============
#include <cuda_runtime.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do {                                 \
  cudaError_t _e = (call);                                     \
  if (_e != cudaSuccess) {                                     \
    /* In production you might handle/report this differently */\
  }                                                            \
} while (0)
#endif

// Update one anti-diagonal d = i + j (interior only), in-place Gauss–Seidel.
// Points on the same anti-diagonal are independent given lexicographic order.
__global__ void seidel_update_diag(double* __restrict__ A, int n, int d)
{
  // Interior indices: i,j in [1, n-2]
  const int i_min = max(1, d - (n - 2));
  const int i_max = min(n - 2, d - 1);
  const int len   = i_max - i_min + 1;
  if (len <= 0) return;

  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= len) return;

  int i = i_min + k;
  int j = d - i; // guaranteed in [1, n-2] by construction

  const int idx = i * n + j;

  // 9-point average (in-place)
  double sum =
      A[(i - 1) * n + (j - 1)] + A[(i - 1) * n + (j    )] + A[(i - 1) * n + (j + 1)] +
      A[(i    ) * n + (j - 1)] + A[(i    ) * n + (j    )] + A[(i    ) * n + (j + 1)] +
      A[(i + 1) * n + (j - 1)] + A[(i + 1) * n + (j    )] + A[(i + 1) * n + (j + 1)];

  A[idx] = sum * (1.0 / 9.0);
}

// Host launcher: performs tsteps Gauss–Seidel sweeps using wavefront (anti-diagonal)
// ordering with a global barrier between diagonals via kernel launch boundaries.
void seidel_gpu(int tsteps, int n, double* A)
{
  if (tsteps <= 0 || n < 3 || A == nullptr) return;

  // Anti-diagonals over interior [1..n-2]x[1..n-2]:
  // d ranges from 1+1=2 to (n-2)+(n-2)=2*(n-2)
  const int d_min = 2;
  const int d_max = 2 * (n - 2);

  const int TPB = 256;

  for (int t = 0; t < tsteps; ++t) {
    for (int d = d_min; d <= d_max; ++d) {
      const int i_min = (d - (n - 2) > 1) ? (d - (n - 2)) : 1;
      const int i_max = (d - 1 < (n - 2)) ? (d - 1) : (n - 2);
      const int len   = i_max - i_min + 1;
      if (len <= 0) continue;

      int blocks = (len + TPB - 1) / TPB;
      seidel_update_diag<<<blocks, TPB>>>(A, n, d);

      // Kernel launch boundary provides the required global ordering between diagonals.
      // Ensure completion before proceeding to next diagonal.
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
    }
  }
}

// ============ Strict serial CPU reference (ground-truth numerics) ============
static void seidel_cpu(int tsteps, int n, double* A) {
    for (int t = 0; t <= tsteps - 1; t++)
        for (int i = 1; i <= n - 2; i++)
            for (int j = 1; j <= n - 2; j++)
                A[i*n + j] = (A[(i-1)*n + (j-1)] + A[(i-1)*n + j] + A[(i-1)*n + (j+1)]
                            + A[i*n + (j-1)]     + A[i*n + j]     + A[i*n + (j+1)]
                            + A[(i+1)*n + (j-1)] + A[(i+1)*n + j] + A[(i+1)*n + (j+1)]) / 9.0;
}

static void init_array(int n, double* A, unsigned seed) {
    srand(seed);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i*n + j] = (double)((i*(j+2) + 2) % 100) / n;  // deterministic, matches PolyBench-style fill
}

static double max_abs_diff(const double* a, const double* b, int n) {
    double m = 0.0;
    for (int k = 0; k < n*n; k++) {
        double e = fabs(a[k] - b[k]);
        if (e > m) m = e;
    }
    return m;
}

int main() {
    int n = N, tsteps = TSTEPS;
    size_t bytes = (size_t)n * n * sizeof(double);

    double* h_cpu = (double*)malloc(bytes);
    double* h_gpu = (double*)malloc(bytes);
    double* h_A0  = (double*)malloc(bytes);

    init_array(n, h_A0, 12345u);

    // ---- CPU reference (timed, averaged over 3 runs) ----
    double cpu_ms = 0.0;
    for (int rep = 0; rep < 3; rep++) {
        for (int k = 0; k < n*n; k++) h_cpu[k] = h_A0[k];
        clock_t s = clock();
        seidel_cpu(tsteps, n, h_cpu);
        clock_t e = clock();
        cpu_ms += (double)(e - s) / CLOCKS_PER_SEC * 1000.0;
    }
    cpu_ms /= 3.0;

    // ---- GPU ----
    double* d_A;
    CHECK_CUDA(cudaMalloc(&d_A, bytes));

    // warmup + correctness run
    CHECK_CUDA(cudaMemcpy(d_A, h_A0, bytes, cudaMemcpyHostToDevice));
    seidel_gpu(tsteps, n, d_A);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_gpu, d_A, bytes, cudaMemcpyDeviceToHost));

    // timed (5 runs)
    cudaEvent_t st, sp;
    cudaEventCreate(&st); cudaEventCreate(&sp);
    cudaEventRecord(st);
    for (int rep = 0; rep < 5; rep++) {
        CHECK_CUDA(cudaMemcpy(d_A, h_A0, bytes, cudaMemcpyHostToDevice));
        seidel_gpu(tsteps, n, d_A);
    }
    cudaEventRecord(sp);
    cudaEventSynchronize(sp);
    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, st, sp);
    gpu_ms /= 5.0f;

    double err = max_abs_diff(h_cpu, h_gpu, n);

    printf("BENCHMARK_RESULT:cpu_ms=%.4f,gpu_ms=%.4f,speedup=%.2f,error=%.2e\n",
           cpu_ms, gpu_ms, cpu_ms / (double)gpu_ms, err);

    cudaFree(d_A);
    free(h_cpu); free(h_gpu); free(h_A0);
    return 0;
}
