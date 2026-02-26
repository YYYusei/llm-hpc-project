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

  // Read nnz count (read-only)
  int nnz = __ldg(&nnz_per_row[row]);

  // Base pointers for this row in ELL/CSR-like fixed-stride storage
  int base = row * max_nnz;

  double sum = 0.0;

  // Unroll for the common 27-pt stencil case; fall back to generic loop otherwise.
  if (max_nnz == 27) {
    #pragma unroll
    for (int j = 0; j < 27; ++j) {
      if (j < nnz) {
        int c = __ldg(&col_ind[base + j]);
        double a = __ldg(&values[base + j]);
        double xv = __ldg(&x[c]);
        sum = fma(a, xv, sum);
      }
    }
  } else {
    for (int j = 0; j < nnz; ++j) {
      int c = __ldg(&col_ind[base + j]);
      double a = __ldg(&values[base + j]);
      double xv = __ldg(&x[c]);
      sum = fma(a, xv, sum);
    }
  }

  y[row] = sum;
}