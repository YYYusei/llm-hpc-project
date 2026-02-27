// Fixed kernel here
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

extern "C" __global__
void symgs_gpu_gpt_5_2(
    int nrow,
    int max_nnz,
    const int* __restrict__ nnz_per_row,
    const int* __restrict__ col_ind,
    const double* __restrict__ values,
    const double* __restrict__ diag,
    const double* __restrict__ r,
    double* __restrict__ x
) {
  // This entry point matches the benchmark's call signature (8 args).
  // To correctly handle Gauss-Seidel dependencies without extra coloring inputs,
  // we perform a strictly ordered (serial) SYMGS sweep on a single thread.
  // This is correct but not parallel; it preserves true GS dependencies.

  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  // Forward sweep: i = 0..nrow-1
  for (int i = 0; i < nrow; i++) {
    double sum = r[i];
    int row_off = i * max_nnz;
    int nnz = nnz_per_row[i];

    #pragma unroll 1
    for (int j = 0; j < nnz; j++) {
      int col = col_ind[row_off + j];
      double a = values[row_off + j];
      sum -= a * x[col];
    }

    double d = diag[i];
    sum += x[i] * d;   // add back diagonal term
    x[i] = sum / d;
  }

  // Backward sweep: i = nrow-1..0
  for (int i = nrow - 1; i >= 0; i--) {
    double sum = r[i];
    int row_off = i * max_nnz;
    int nnz = nnz_per_row[i];

    #pragma unroll 1
    for (int j = 0; j < nnz; j++) {
      int col = col_ind[row_off + j];
      double a = values[row_off + j];
      sum -= a * x[col];
    }

    double d = diag[i];
    sum += x[i] * d;
    x[i] = sum / d;
  }
}