__global__ void symgs_forward_sweep(const int *row_ptr, const int *col_ind, const double *values, const double *diag, const double *r, double *x, const int *levels, const int *level_ptr, int num_levels) {
    int level = blockIdx.x;
    if (level >= num_levels) return;

    int start = level_ptr[level];
    int end = level_ptr[level + 1];

    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        int row = levels[i];
        double sum = r[row];
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];

        for (int j = row_start; j < row_end; j++) {
            int col = col_ind[j];
            if (col < row) {
                sum -= values[j] * x[col];
            }
        }
        x[row] = sum / diag[row];
    }
}

__global__ void symgs_backward_sweep(const int *row_ptr, const int *col_ind, const double *values, const double *diag, const double *r, double *x, const int *levels, const int *level_ptr, int num_levels) {
    int level = blockIdx.x;
    if (level >= num_levels) return;

    int start = level_ptr[level];
    int end = level_ptr[level + 1];

    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        int row = levels[i];
        double sum = r[row];
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];

        for (int j = row_start; j < row_end; j++) {
            int col = col_ind[j];
            if (col > row) {
                sum -= values[j] * x[col];
            }
        }
        x[row] = sum / diag[row];
    }
}

void symgs_gpu_gpt_4o(const int *row_ptr, const int *col_ind, const double *values, const double *diag, const double *r, double *x, const int *levels, const int *level_ptr, int num_levels) {
    int threads_per_block = 256;
    int blocks_per_grid = num_levels;

    // Forward sweep
    symgs_forward_sweep<<<blocks_per_grid, threads_per_block>>>(row_ptr, col_ind, values, diag, r, x, levels, level_ptr, num_levels);
    cudaDeviceSynchronize();

    // Backward sweep
    symgs_backward_sweep<<<blocks_per_grid, threads_per_block>>>(row_ptr, col_ind, values, diag, r, x, levels, level_ptr, num_levels);
    cudaDeviceSynchronize();
}