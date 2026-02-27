// Your multi-colored SYMGS implementation

#include <cuda_runtime.h>

__global__ void symgs_kernel(int nrow, int max_nnz, int target_color,
    const int* row_colors, const int* nnz_per_row, const int* col_ind,
    const double* values, const double* diag, const double* r, double* x)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nrow) return;
  if (row_colors[i] != target_color) return;

  const int row_start = i * max_nnz;
  const int nnz = nnz_per_row[i];
  const double d = diag[i];

  double sum = r[i];

  #pragma unroll
  for (int j = 0; j < 27; j++) { // max_nnz is 27 per problem statement
    if (j >= nnz) break;
    const int col = col_ind[row_start + j];
    const double aij = values[row_start + j];
    sum -= aij * x[col];
  }

  // Add back diagonal contribution (since loop subtracts Aii*x[i] too)
  sum += x[i] * d;

  x[i] = sum / d;
}

void symgs_gpu_gpt_5_2(int nrow, int max_nnz, int num_colors,
    const int* row_colors, const int* nnz_per_row, const int* col_ind,
    const double* values, const double* diag, const double* r, double* x)
{
  const int block = 256;
  const int grid  = (nrow + block - 1) / block;

  // Forward sweep: colors 0..num_colors-1
  for (int c = 0; c < num_colors; c++) {
    symgs_kernel<<<grid, block>>>(nrow, max_nnz, c,
                                  row_colors, nnz_per_row, col_ind,
                                  values, diag, r, x);
    cudaDeviceSynchronize();
  }

  // Backward sweep: colors num_colors-1..0
  for (int c = num_colors - 1; c >= 0; c--) {
    symgs_kernel<<<grid, block>>>(nrow, max_nnz, c,
                                  row_colors, nnz_per_row, col_ind,
                                  values, diag, r, x);
    cudaDeviceSynchronize();
  }
}