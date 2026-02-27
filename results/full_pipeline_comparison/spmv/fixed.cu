#include <stdint.h>

__global__ void spmv_pipeline(int nrow, int /*max_nnz*/,
    const int* __restrict__ nnz_per_row, const int* __restrict__ col_ind,
    const double* __restrict__ values, const double* __restrict__ x, double* __restrict__ y)
{
  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  const int warps_per_block = blockDim.x >> 5;
  const int row = blockIdx.x * warps_per_block + warp_in_block;

  if (row >= nrow) return;

  int start = 0;
#pragma unroll 1
  for (int r = 0; r < row; ++r) start += __ldg(&nnz_per_row[r]);

  const int nnz = __ldg(&nnz_per_row[row]);
  const int end = start + nnz;

  double sum = 0.0;

  int jj = start + lane;

  const uintptr_t vptr0 = (uintptr_t)(values + jj);
  const uintptr_t iptr0 = (uintptr_t)(col_ind + jj);
  const bool can_vec2 = ((vptr0 & 0xF) == 0) && ((iptr0 & 0x7) == 0);

  if (can_vec2) {
    for (; jj + 32 < end; jj += 64) {
      const double2 a2 = *reinterpret_cast<const double2*>(values + jj);
      const int2    c2 = *reinterpret_cast<const int2*>(col_ind + jj);

      const double x0 = __ldg(&x[c2.x]);
      const double x1 = __ldg(&x[c2.y]);

      sum = fma(a2.x, x0, sum);
      sum = fma(a2.y, x1, sum);
    }
  }

  for (; jj < end; jj += 32) {
    const int col = __ldg(&col_ind[jj]);
    const double a = __ldg(&values[jj]);
    const double xv = __ldg(&x[col]);
    sum = fma(a, xv, sum);
  }

  sum += __shfl_down_sync(0xffffffff, sum, 16);
  sum += __shfl_down_sync(0xffffffff, sum, 8);
  sum += __shfl_down_sync(0xffffffff, sum, 4);
  sum += __shfl_down_sync(0xffffffff, sum, 2);
  sum += __shfl_down_sync(0xffffffff, sum, 1);

  if (lane == 0) y[row] = sum;
}