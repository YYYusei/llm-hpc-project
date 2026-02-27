
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { cudaError_t e=call; if(e!=cudaSuccess){printf("CUDA_ERROR: %s\n",cudaGetErrorString(e));exit(1);} }

// ============ LLM Generated SYMGS Kernel(s) ============
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

// ============ CPU Reference (Sequential SYMGS) ============
void symgs_cpu(int nrow, int max_nnz, const int* nnz_per_row,
               const int* col_ind, const double* values, const double* diag,
               const double* r, double* x) {
    // Forward sweep
    for (int i = 0; i < nrow; i++) {
        double sum = r[i];
        int row_nnz = nnz_per_row[i];
        for (int j = 0; j < row_nnz; j++) {
            int idx = i * max_nnz + j;
            int col = col_ind[idx];
            sum -= values[idx] * x[col];
        }
        sum += x[i] * diag[i];
        x[i] = sum / diag[i];
    }
    
    // Backward sweep
    for (int i = nrow - 1; i >= 0; i--) {
        double sum = r[i];
        int row_nnz = nnz_per_row[i];
        for (int j = 0; j < row_nnz; j++) {
            int idx = i * max_nnz + j;
            int col = col_ind[idx];
            sum -= values[idx] * x[col];
        }
        sum += x[i] * diag[i];
        x[i] = sum / diag[i];
    }
}

double check_correctness(double* x1, double* x2, int n) {
    double maxerr = 0.0;
    for (int i = 0; i < n; i++) {
        double err = fabs(x1[i] - x2[i]);
        if (err > maxerr) maxerr = err;
    }
    return maxerr;
}

int main() {
    // Problem size: smaller for SYMGS due to dependencies
    int nrow = 50000;
    int max_nnz = 27;
    
    // Allocate host memory
    int* h_nnz_per_row = (int*)malloc(nrow * sizeof(int));
    int* h_col_ind = (int*)malloc(nrow * max_nnz * sizeof(int));
    double* h_values = (double*)malloc(nrow * max_nnz * sizeof(double));
    double* h_diag = (double*)malloc(nrow * sizeof(double));
    double* h_r = (double*)malloc(nrow * sizeof(double));
    double* h_x_cpu = (double*)malloc(nrow * sizeof(double));
    double* h_x_gpu = (double*)malloc(nrow * sizeof(double));
    
    // Initialize data (simulate 3D 27-point stencil)
    srand(12345);
    for (int i = 0; i < nrow; i++) {
        h_nnz_per_row[i] = 27;
        h_diag[i] = 26.0;  // Diagonal dominant
        h_r[i] = (double)rand() / RAND_MAX;
        h_x_cpu[i] = 0.0;
        h_x_gpu[i] = 0.0;
        
        for (int j = 0; j < max_nnz; j++) {
            int idx = i * max_nnz + j;
            if (j == 13) {  // Diagonal position
                h_col_ind[idx] = i;
                h_values[idx] = 26.0;
            } else {
                int offset = (j < 13) ? (j - 13) : (j - 13);
                int col = i + offset * 100 + (rand() % 10 - 5);
                if (col < 0) col = 0;
                if (col >= nrow) col = nrow - 1;
                h_col_ind[idx] = col;
                h_values[idx] = -1.0;
            }
        }
    }
    
    // CPU benchmark
    clock_t cpu_start = clock();
    for (int iter = 0; iter < 3; iter++) {
        for (int i = 0; i < nrow; i++) h_x_cpu[i] = 0.0;
        symgs_cpu(nrow, max_nnz, h_nnz_per_row, h_col_ind, h_values, h_diag, h_r, h_x_cpu);
    }
    clock_t cpu_end = clock();
    double cpu_ms = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000.0 / 3.0;
    
    // Allocate device memory
    int *d_nnz_per_row, *d_col_ind;
    double *d_values, *d_diag, *d_r, *d_x;
    CHECK_CUDA(cudaMalloc(&d_nnz_per_row, nrow * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_col_ind, nrow * max_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_values, nrow * max_nnz * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_diag, nrow * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_r, nrow * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_x, nrow * sizeof(double)));
    
    CHECK_CUDA(cudaMemcpy(d_nnz_per_row, h_nnz_per_row, nrow * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_ind, h_col_ind, nrow * max_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_values, h_values, nrow * max_nnz * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_diag, h_diag, nrow * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_r, h_r, nrow * sizeof(double), cudaMemcpyHostToDevice));
    
    int bs = 256, nb = (nrow + bs - 1) / bs;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    // GPU benchmark
    CHECK_CUDA(cudaMemset(d_x, 0, nrow * sizeof(double)));
    symgs_gpu_gpt_4o(nrow, max_nnz, d_nnz_per_row, d_col_ind, d_values, d_diag, d_r, d_x);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int iter = 0; iter < 5; iter++) {
        CHECK_CUDA(cudaMemset(d_x, 0, nrow * sizeof(double)));
        symgs_gpu_gpt_4o(nrow, max_nnz, d_nnz_per_row, d_col_ind, d_values, d_diag, d_r, d_x);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, start, stop);
    gpu_ms /= 5;
    
    CHECK_CUDA(cudaMemcpy(h_x_gpu, d_x, nrow * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Check correctness (note: may have some error due to different ordering)
    // Reset CPU result for fair comparison
    for (int i = 0; i < nrow; i++) h_x_cpu[i] = 0.0;
    symgs_cpu(nrow, max_nnz, h_nnz_per_row, h_col_ind, h_values, h_diag, h_r, h_x_cpu);
    
    double err = check_correctness(h_x_cpu, h_x_gpu, nrow);
    
    printf("BENCHMARK_RESULT:cpu_ms=%.4f,gpu_ms=%.4f,speedup=%.2f,error=%.2e\n",
           cpu_ms, gpu_ms, cpu_ms/gpu_ms, err);
    
    // Cleanup
    free(h_nnz_per_row); free(h_col_ind); free(h_values);
    free(h_diag); free(h_r); free(h_x_cpu); free(h_x_gpu);
    cudaFree(d_nnz_per_row); cudaFree(d_col_ind); cudaFree(d_values);
    cudaFree(d_diag); cudaFree(d_r); cudaFree(d_x);
    
    return 0;
}
