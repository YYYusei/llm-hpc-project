__global__ void spmv_kernel_gpt_4o(const double* __restrict__ values, 
                                   const int* __restrict__ col_ind, 
                                   const double* __restrict__ x, 
                                   double* __restrict__ y, 
                                   const int* __restrict__ nnz_per_row, 
                                   const int nrow, 
                                   const int max_nnz) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrow) {
        double sum = 0.0;
        int row_start = row * max_nnz;
        int nnz = nnz_per_row[row];
        
        for (int j = 0; j < nnz; j++) {
            int col = __ldg(&col_ind[row_start + j]);
            double val = __ldg(&values[row_start + j]);
            sum += val * __ldg(&x[col]);
        }
        
        y[row] = sum;
    }
}