__global__ void spmv_pipeline(int nrow, int /*max_nnz*/,
    const int* __restrict__ nnz_per_row, const int* __restrict__ col_ind,
    const double* __restrict__ values, const double* __restrict__ x, double* __restrict__ y)
{
  // Warp-per-row CSR-vector style kernel, assuming rows are stored contiguously:
  // row r occupies [rowPtr(r), rowPtr(r)+nnz_per_row[r]) in col_ind/values,
  // where rowPtr(r) = exclusive prefix sum of nnz_per_row (computed offline).
  //
  // NOTE: This kernel uses nnz_per_row to derive row boundaries; it assumes
  // col_ind/values are laid out in a flattened-by-row order.

  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  const int warps_per_block = blockDim.x >> 5;
  const int row = blockIdx.x * warps_per_block + warp_in_block;

  if (row >= nrow) return;

  // Compute row start via prefix sum over nnz_per_row.
  // For performance, nnz_per_row should be converted to rowPtr on the host/device
  // and passed instead; kept as-is to satisfy required signature.
  int start = 0;
#pragma unroll 1
  for (int r = 0; r < row; ++r) start += __ldg(&nnz_per_row[r]);

  const int nnz = __ldg(&nnz_per_row[row]);
  const int end = start + nnz;

  double sum = 0.0;

  // Vectorized loads for values/col_ind when aligned and lane==0..31 stepping by warp.
  // We process 2 entries per iteration per lane when possible.
  int jj = start + lane;

  // Main loop: try double2/int2 vectorized loads when 8-byte aligned for values and col_ind.
  // Alignment check is uniform across the warp for a given jj pattern.
  const uintptr_t vptr0 = (uintptr_t)(values + jj);
  const uintptr_t iptr0 = (uintptr_t)(col_ind + jj);
  const bool can_vec2 = ((vptr0 & 0xF) == 0) && ((iptr0 & 0x7) == 0); // double2 needs 16B, int2 needs 8B

  if (can_vec2) {
    for (; jj + 32 < end; jj += 64) { // each lane handles 2 items: jj and jj+32
      // Load two values and two indices (coalesced across warp)
      const double2 a2 = *reinterpret_cast<const double2 const*>(values + jj);
      const int2    c2 = *reinterpret_cast<const int2 const*>(col_ind + jj);

      const double x0 = __ldg(&x[c2.x]);
      const double x1 = __ldg(&x[c2.y]);

      sum = fma(a2.x, x0, sum);
      sum = fma(a2.y, x1, sum);
    }
  }

  // Remainder / fallback scalar path
  for (; jj < end; jj += 32) {
    const int col = __ldg(&col_ind[jj]);
    const double a = __ldg(&values[jj]);
    const double xv = __ldg(&x[col]);
    sum = fma(a, xv, sum);
  }

  // Warp reduction (no shared memory)
  sum += __shfl_down_sync(0xffffffff, sum, 16);
  sum += __shfl_down_sync(0xffffffff, sum, 8);
  sum += __shfl_down_sync(0xffffffff, sum, 4);
  sum += __shfl_down_sync(0xffffffff, sum, 2);
  sum += __shfl_down_sync(0xffffffff, sum, 1);

  if (lane == 0) y[row] = sum;
}