extern "C" __global__
void spmv_kernel_gpt_5_2(const int nrow,
                         const int max_nnz,
                         const int *__restrict__ nnz_per_row,
                         const int *__restrict__ col_ind,
                         const double *__restrict__ values,
                         const double *__restrict__ x,
                         double *__restrict__ y) {
  int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= nrow) return;

  const int nnz = __ldg(&nnz_per_row[row]);
  const int base = row * max_nnz;

  double sum = 0.0;

  // Unroll for the common 27-pt stencil case; fall back to generic loop otherwise.
  if (nnz == 27) {
#pragma unroll
    for (int j = 0; j < 27; ++j) {
      const int c = __ldg(&col_ind[base + j]);
      const double a = __ldg(&values[base + j]);
      const double xv = __ldg(&x[c]);
      sum = fma(a, xv, sum);
    }
  } else {
    for (int j = 0; j < nnz; ++j) {
      const int c = __ldg(&col_ind[base + j]);
      const double a = __ldg(&values[base + j]);
      const double xv = __ldg(&x[c]);
      sum = fma(a, xv, sum);
    }
  }

  y[row] = sum;
}