// Fixed kernel here

// NOTE: This implementation matches the benchmark call signature:
//   symgs_gpu_gpt_4o(nrow, max_nnz, d_nnz_per_row, d_col_ind, d_values, d_diag, d_r, d_x);
//
// It also enforces Gauss-Seidel data dependencies by running one CUDA block per row
// and processing rows sequentially on the device (single grid, single block).
// This is correct but not highly parallel.

__global__ void symgs_forward_sweep_rowserial(
    int nrow, int max_nnz,
    const int* __restrict__ nnz_per_row,
    const int* __restrict__ col_ind,
    const double* __restrict__ values,
    const double* __restrict__ diag,
    const double* __restrict__ r,
    double* __restrict__ x)
{
    // Enforce strict row order for forward sweep
    for (int row = 0; row < nrow; ++row) {
        __syncthreads();

        // Compute row base in ELLPACK-like layout: row-major with stride max_nnz
        int base = row * max_nnz;
        int nnz  = nnz_per_row[row];

        // Parallel reduction of sum over this row within the block
        double local = 0.0;
        for (int j = threadIdx.x; j < nnz; j += blockDim.x) {
            int idx = base + j;
            int col = col_ind[idx];
            double a = values[idx];

            if (col != row) {
                local += a * x[col];
            }
        }

        // Block reduction
        __shared__ double sh[256]; // assumes blockDim.x <= 256 as used by host
        sh[threadIdx.x] = local;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (threadIdx.x < offset) sh[threadIdx.x] += sh[threadIdx.x + offset];
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            double sum = r[row] - sh[0];
            x[row] = sum / diag[row];
        }
        __syncthreads();
    }
}

__global__ void symgs_backward_sweep_rowserial(
    int nrow, int max_nnz,
    const int* __restrict__ nnz_per_row,
    const int* __restrict__ col_ind,
    const double* __restrict__ values,
    const double* __restrict__ diag,
    const double* __restrict__ r,
    double* __restrict__ x)
{
    // Enforce strict reverse row order for backward sweep
    for (int row = nrow - 1; row >= 0; --row) {
        __syncthreads();

        int base = row * max_nnz;
        int nnz  = nnz_per_row[row];

        double local = 0.0;
        for (int j = threadIdx.x; j < nnz; j += blockDim.x) {
            int idx = base + j;
            int col = col_ind[idx];
            double a = values[idx];

            if (col != row) {
                local += a * x[col];
            }
        }

        __shared__ double sh[256]; // assumes blockDim.x <= 256 as used by host
        sh[threadIdx.x] = local;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (threadIdx.x < offset) sh[threadIdx.x] += sh[threadIdx.x + offset];
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            double sum = r[row] - sh[0];
            x[row] = sum / diag[row];
        }
        __syncthreads();
    }
}

void symgs_gpu_gpt_4o(
    int nrow, int max_nnz,
    const int* d_nnz_per_row,
    const int* d_col_ind,
    const double* d_values,
    const double* d_diag,
    const double* d_r,
    double* d_x)
{
    // One block to preserve strict GS dependencies; 256 threads as in original code.
    dim3 block(256);
    dim3 grid(1);

    symgs_forward_sweep_rowserial<<<grid, block>>>(
        nrow, max_nnz, d_nnz_per_row, d_col_ind, d_values, d_diag, d_r, d_x);
    cudaDeviceSynchronize();

    symgs_backward_sweep_rowserial<<<grid, block>>>(
        nrow, max_nnz, d_nnz_per_row, d_col_ind, d_values, d_diag, d_r, d_x);
    cudaDeviceSynchronize();
}