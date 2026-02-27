
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { cudaError_t e=call; if(e!=cudaSuccess){printf("CUDA_ERROR: %s\n",cudaGetErrorString(e));exit(1);} }

// ============ LLM Generated SYMGS ============
// kernel + host function
#include <cuda_runtime.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do {                                 \
  cudaError_t err__ = (call);                                  \
  if (err__ != cudaSuccess) {                                  \
    /* In production, replace with proper error handling. */   \
  }                                                            \
} while (0)
#endif

__global__ void symgs_color_kernel(
    int nrow, int max_nnz, int color,
    const int* __restrict__ row_colors,
    const int* __restrict__ nnz_per_row,
    const int* __restrict__ col_ind,
    const double* __restrict__ values,
    const double* __restrict__ diag,
    const double* __restrict__ r,
    double* __restrict__ x)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = tid; i < nrow; i += stride) {
    if (row_colors[i] != color) continue;

    const double di = diag[i];
    // Assume di != 0.0
    double sum = r[i];

    const int base = i * max_nnz;
    const int nnz  = nnz_per_row[i];

    // sum -= A[i,*] * x[*]
    #pragma unroll 1
    for (int j = 0; j < nnz; ++j) {
      const int col = col_ind[base + j];
      const double a = values[base + j];
      sum -= a * x[col];
    }

    // Add back diagonal contribution (since it was subtracted above)
    sum += x[i] * di;

    // Gauss-Seidel update
    x[i] = sum / di;
  }
}

void symgs_gpu_pipeline(int nrow, int max_nnz, int num_colors,
    const int* row_colors, const int* nnz_per_row, const int* col_ind,
    const double* values, const double* diag, const double* r, double* x)
{
  // Assumes all pointers are device pointers and data is already on GPU.

  const int block = 256;
  int grid = (nrow + block - 1) / block;
  // Cap grid to a reasonable size to avoid excessive launch overhead.
  grid = (grid > 65535) ? 65535 : grid;

  // Forward sweep: colors 0 .. num_colors-1
  for (int c = 0; c < num_colors; ++c) {
    symgs_color_kernel<<<grid, block>>>(
        nrow, max_nnz, c,
        row_colors, nnz_per_row, col_ind, values, diag, r, x);
  }

  // Backward sweep: colors num_colors-1 .. 0
  for (int c = num_colors - 1; c >= 0; --c) {
    symgs_color_kernel<<<grid, block>>>(
        nrow, max_nnz, c,
        row_colors, nnz_per_row, col_ind, values, diag, r, x);
  }

  // Ensure completion before returning (optional depending on caller semantics).
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

// ============ CPU Reference ============
void symgs_cpu(int nrow, int max_nnz, const int* nnz_per_row,
               const int* col_ind, const double* values, const double* diag,
               const double* r, double* x) {
    for (int i = 0; i < nrow; i++) {
        double sum = r[i];
        for (int j = 0; j < nnz_per_row[i]; j++) {
            int idx = i * max_nnz + j;
            sum -= values[idx] * x[col_ind[idx]];
        }
        sum += x[i] * diag[i];
        x[i] = sum / diag[i];
    }
    for (int i = nrow - 1; i >= 0; i--) {
        double sum = r[i];
        for (int j = 0; j < nnz_per_row[i]; j++) {
            int idx = i * max_nnz + j;
            sum -= values[idx] * x[col_ind[idx]];
        }
        sum += x[i] * diag[i];
        x[i] = sum / diag[i];
    }
}

void compute_colors(int nrow, int* row_colors, int num_colors) {
    int nx = 50, ny = 50, nz = nrow / (50 * 50);
    if (nz < 1) nz = 1;
    for (int i = 0; i < nrow; i++) {
        int iz = i / (nx * ny);
        int iy = (i % (nx * ny)) / nx;
        int ix = i % nx;
        row_colors[i] = (ix + iy + iz) % num_colors;
    }
}

double check_correctness(double* x1, double* x2, int n) {
    double maxerr = 0.0;
    for (int i = 0; i < n; i++) {
        double err = fabs(x1[i] - x2[i]);
        if (err > maxerr) maxerr = err;
    }
    return maxerr;
}

int main() {
    int nrow = 50000;
    int max_nnz = 27;
    int num_colors = 8;
    
    int* h_nnz_per_row = (int*)malloc(nrow * sizeof(int));
    int* h_col_ind = (int*)malloc(nrow * max_nnz * sizeof(int));
    double* h_values = (double*)malloc(nrow * max_nnz * sizeof(double));
    double* h_diag = (double*)malloc(nrow * sizeof(double));
    double* h_r = (double*)malloc(nrow * sizeof(double));
    double* h_x_cpu = (double*)malloc(nrow * sizeof(double));
    double* h_x_gpu = (double*)malloc(nrow * sizeof(double));
    int* h_row_colors = (int*)malloc(nrow * sizeof(int));
    
    compute_colors(nrow, h_row_colors, num_colors);
    
    srand(12345);
    for (int i = 0; i < nrow; i++) {
        h_nnz_per_row[i] = 27;
        h_diag[i] = 26.0;
        h_r[i] = (double)rand() / RAND_MAX;
        h_x_cpu[i] = 0.0;
        h_x_gpu[i] = 0.0;
        for (int j = 0; j < max_nnz; j++) {
            int idx = i * max_nnz + j;
            if (j == 13) {
                h_col_ind[idx] = i;
                h_values[idx] = 26.0;
            } else {
                int offset = (j < 13) ? (j - 13) : (j - 13);
                int col = i + offset * 100 + (rand() % 10 - 5);
                if (col < 0) col = 0;
                if (col >= nrow) col = nrow - 1;
                h_col_ind[idx] = col;
                h_values[idx] = -1.0;
            }
        }
    }
    
    clock_t cpu_start = clock();
    for (int iter = 0; iter < 3; iter++) {
        for (int i = 0; i < nrow; i++) h_x_cpu[i] = 0.0;
        symgs_cpu(nrow, max_nnz, h_nnz_per_row, h_col_ind, h_values, h_diag, h_r, h_x_cpu);
    }
    clock_t cpu_end = clock();
    double cpu_ms = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000.0 / 3.0;
    
    int *d_nnz_per_row, *d_col_ind, *d_row_colors;
    double *d_values, *d_diag, *d_r, *d_x;
    CHECK_CUDA(cudaMalloc(&d_nnz_per_row, nrow * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_col_ind, nrow * max_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_values, nrow * max_nnz * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_diag, nrow * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_r, nrow * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_x, nrow * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_row_colors, nrow * sizeof(int)));
    
    CHECK_CUDA(cudaMemcpy(d_nnz_per_row, h_nnz_per_row, nrow * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_ind, h_col_ind, nrow * max_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_values, h_values, nrow * max_nnz * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_diag, h_diag, nrow * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_r, h_r, nrow * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_row_colors, h_row_colors, nrow * sizeof(int), cudaMemcpyHostToDevice));
    
    CHECK_CUDA(cudaMemset(d_x, 0, nrow * sizeof(double)));
    symgs_gpu_pipeline(nrow, max_nnz, num_colors, d_row_colors, d_nnz_per_row, d_col_ind, d_values, d_diag, d_r, d_x);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int iter = 0; iter < 5; iter++) {
        CHECK_CUDA(cudaMemset(d_x, 0, nrow * sizeof(double)));
        symgs_gpu_pipeline(nrow, max_nnz, num_colors, d_row_colors, d_nnz_per_row, d_col_ind, d_values, d_diag, d_r, d_x);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, start, stop);
    gpu_ms /= 5;
    
    CHECK_CUDA(cudaMemcpy(h_x_gpu, d_x, nrow * sizeof(double), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < nrow; i++) h_x_cpu[i] = 0.0;
    symgs_cpu(nrow, max_nnz, h_nnz_per_row, h_col_ind, h_values, h_diag, h_r, h_x_cpu);
    double err = check_correctness(h_x_cpu, h_x_gpu, nrow);
    
    printf("BENCHMARK_RESULT:cpu_ms=%.4f,gpu_ms=%.4f,speedup=%.2f,error=%.2e\n",
           cpu_ms, gpu_ms, cpu_ms/gpu_ms, err);
    
    return 0;
}
