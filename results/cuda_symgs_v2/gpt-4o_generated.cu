// Your multi-colored SYMGS implementation

#include <cuda_runtime.h>

__global__ void symgs_kernel(int nrow, int max_nnz, int target_color,
    const int* row_colors, const int* nnz_per_row, const int* col_ind,
    const double* values, const double* diag, const double* r, double* x) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nrow) return;

    if (row_colors[i] == target_color) {
        double sum = r[i];
        int row_start = i * max_nnz;
        int nnz = nnz_per_row[i];
        for (int j = 0; j < nnz; j++) {
            int col = col_ind[row_start + j];
            sum -= values[row_start + j] * x[col];
        }
        sum += x[i] * diag[i];
        x[i] = sum / diag[i];
    }
}

void symgs_gpu_gpt_4o(int nrow, int max_nnz, int num_colors,
    const int* row_colors, const int* nnz_per_row, const int* col_ind,
    const double* values, const double* diag, const double* r, double* x) {

    int blockSize = 256;
    int numBlocks = (nrow + blockSize - 1) / blockSize;

    // Forward sweep
    for (int c = 0; c < num_colors; c++) {
        symgs_kernel<<<numBlocks, blockSize>>>(nrow, max_nnz, c, row_colors, nnz_per_row, col_ind, values, diag, r, x);
        cudaDeviceSynchronize();
    }

    // Backward sweep
    for (int c = num_colors - 1; c >= 0; c--) {
        symgs_kernel<<<numBlocks, blockSize>>>(nrow, max_nnz, c, row_colors, nnz_per_row, col_ind, values, diag, r, x);
        cudaDeviceSynchronize();
    }
}