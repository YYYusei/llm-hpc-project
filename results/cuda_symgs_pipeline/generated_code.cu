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