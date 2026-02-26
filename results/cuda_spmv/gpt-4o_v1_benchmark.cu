
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { cudaError_t e=call; if(e!=cudaSuccess){printf("CUDA_ERROR: %s\n",cudaGetErrorString(e));exit(1);} }

// ============ Optimized SPMV Kernel ============
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

// ============ CPU Reference ============
void spmv_cpu(int nrow, int max_nnz, const int* nnz_per_row, 
              const int* col_ind, const double* values,
              const double* x, double* y) {
    for (int i = 0; i < nrow; i++) {
        double sum = 0.0;
        int row_nnz = nnz_per_row[i];
        for (int j = 0; j < row_nnz; j++) {
            int idx = i * max_nnz + j;
            sum += values[idx] * x[col_ind[idx]];
        }
        y[i] = sum;
    }
}

double check_correctness(double* y1, double* y2, int n) {
    double maxerr = 0.0;
    for (int i = 0; i < n; i++) {
        double err = fabs(y1[i] - y2[i]);
        if (err > maxerr) maxerr = err;
    }
    return maxerr;
}

int main() {
    // Problem size: 100K rows, ~27 non-zeros per row (3D stencil)
    int nrow = 100000;
    int max_nnz = 27;
    int ncol = nrow;  // Square matrix
    
    // Allocate host memory
    int* h_nnz_per_row = (int*)malloc(nrow * sizeof(int));
    int* h_col_ind = (int*)malloc(nrow * max_nnz * sizeof(int));
    double* h_values = (double*)malloc(nrow * max_nnz * sizeof(double));
    double* h_x = (double*)malloc(ncol * sizeof(double));
    double* h_y_cpu = (double*)malloc(nrow * sizeof(double));
    double* h_y_gpu = (double*)malloc(nrow * sizeof(double));
    
    // Initialize data (simulate 3D 27-point stencil)
    srand(12345);
    for (int i = 0; i < nrow; i++) {
        h_nnz_per_row[i] = 27;  // Fixed for stencil
        for (int j = 0; j < max_nnz; j++) {
            int idx = i * max_nnz + j;
            // Diagonal-dominant pattern
            if (j == 13) {  // Center point
                h_col_ind[idx] = i;
                h_values[idx] = 26.0;
            } else {
                // Random neighbor within bounds
                int offset = (j < 13) ? (j - 13) : (j - 13);
                int col = i + offset * 100 + (rand() % 10 - 5);
                if (col < 0) col = 0;
                if (col >= ncol) col = ncol - 1;
                h_col_ind[idx] = col;
                h_values[idx] = -1.0;
            }
        }
    }
    
    for (int i = 0; i < ncol; i++) {
        h_x[i] = (double)rand() / RAND_MAX;
    }
    
    // CPU benchmark
    clock_t cpu_start = clock();
    for (int r = 0; r < 5; r++)
        spmv_cpu(nrow, max_nnz, h_nnz_per_row, h_col_ind, h_values, h_x, h_y_cpu);
    clock_t cpu_end = clock();
    double cpu_ms = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000.0 / 5.0;
    
    // Allocate device memory
    int *d_nnz_per_row, *d_col_ind;
    double *d_values, *d_x, *d_y;
    CHECK_CUDA(cudaMalloc(&d_nnz_per_row, nrow * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_col_ind, nrow * max_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_values, nrow * max_nnz * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_x, ncol * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_y, nrow * sizeof(double)));
    
    CHECK_CUDA(cudaMemcpy(d_nnz_per_row, h_nnz_per_row, nrow * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_ind, h_col_ind, nrow * max_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_values, h_values, nrow * max_nnz * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, ncol * sizeof(double), cudaMemcpyHostToDevice));
    
    int bs = 256, nb = (nrow + bs - 1) / bs;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    // GPU benchmark
    spmv_kernel_gpt_4o<<<nb, bs>>>(nrow, max_nnz, d_nnz_per_row, d_col_ind, d_values, d_x, d_y);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int r = 0; r < 10; r++)
        spmv_kernel_gpt_4o<<<nb, bs>>>(nrow, max_nnz, d_nnz_per_row, d_col_ind, d_values, d_x, d_y);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, start, stop);
    gpu_ms /= 10;
    
    CHECK_CUDA(cudaMemcpy(h_y_gpu, d_y, nrow * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Check correctness
    double err = check_correctness(h_y_cpu, h_y_gpu, nrow);
    
    printf("BENCHMARK_RESULT:cpu_ms=%.4f,gpu_ms=%.4f,speedup=%.2f,error=%.2e\n",
           cpu_ms, gpu_ms, cpu_ms/gpu_ms, err);
    
    // Cleanup
    free(h_nnz_per_row); free(h_col_ind); free(h_values);
    free(h_x); free(h_y_cpu); free(h_y_gpu);
    cudaFree(d_nnz_per_row); cudaFree(d_col_ind); cudaFree(d_values);
    cudaFree(d_x); cudaFree(d_y);
    
    return 0;
}
