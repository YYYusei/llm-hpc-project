__global__ void spmv_kernel_gpt_4o(const int nrow,
                                   const int max_nnz,
                                   const int* __restrict__ nnz_per_row,
                                   const int* __restrict__ col_ind,
                                   const double* __restrict__ values,
                                   const double* __restrict__ x,
                                   double* __restrict__ y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < nrow) {
        double sum = 0.0;
        int row_start = row * max_nnz;
        int nnz = nnz_per_row[row];

        for (int j = 0; j < nnz; j++) {
            int idx = row_start + j;
            int col = __ldg(&col_ind[idx]);
            double val = __ldg(&values[idx]);
            sum += val * __ldg(&x[col]);
        }

        y[row] = sum;
    }
}