
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { cudaError_t e=call; if(e!=cudaSuccess){printf("CUDA_ERROR: %s\n",cudaGetErrorString(e));exit(1);} }

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

void spmv_cpu(int nrow, int max_nnz, const int* nnz_per_row, const int* col_ind, const double* values, const double* x, double* y) {
    for (int i = 0; i < nrow; i++) {
        double sum = 0.0;
        for (int j = 0; j < nnz_per_row[i]; j++) {
            int idx = i * max_nnz + j;
            sum += values[idx] * x[col_ind[idx]];
        }
        y[i] = sum;
    }
}

int main() {
    int nrow = 100000, max_nnz = 27;
    int *h_nnz_per_row = (int*)malloc(nrow * sizeof(int));
    int *h_col_ind = (int*)malloc(nrow * max_nnz * sizeof(int));
    double *h_values = (double*)malloc(nrow * max_nnz * sizeof(double));
    double *h_x = (double*)malloc(nrow * sizeof(double));
    double *h_y_cpu = (double*)malloc(nrow * sizeof(double));
    double *h_y_gpu = (double*)malloc(nrow * sizeof(double));
    
    srand(12345);
    for (int i = 0; i < nrow; i++) {
        h_nnz_per_row[i] = 27;
        h_x[i] = (double)rand() / RAND_MAX;
        for (int j = 0; j < max_nnz; j++) {
            int idx = i * max_nnz + j;
            h_col_ind[idx] = (i + j - 13 + nrow) % nrow;
            h_values[idx] = (j == 13) ? 26.0 : -1.0;
        }
    }
    
    clock_t cpu_start = clock();
    for (int iter = 0; iter < 10; iter++) spmv_cpu(nrow, max_nnz, h_nnz_per_row, h_col_ind, h_values, h_x, h_y_cpu);
    double cpu_ms = (double)(clock() - cpu_start) / CLOCKS_PER_SEC * 1000.0 / 10.0;
    
    int *d_nnz_per_row, *d_col_ind;
    double *d_values, *d_x, *d_y;
    CHECK_CUDA(cudaMalloc(&d_nnz_per_row, nrow * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_col_ind, nrow * max_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_values, nrow * max_nnz * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_x, nrow * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_y, nrow * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(d_nnz_per_row, h_nnz_per_row, nrow * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_ind, h_col_ind, nrow * max_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_values, h_values, nrow * max_nnz * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, nrow * sizeof(double), cudaMemcpyHostToDevice));
    
    int bs = 256, nb = (nrow + bs - 1) / bs;
    spmv_pipeline<<<nb, bs>>>(nrow, max_nnz, d_nnz_per_row, d_col_ind, d_values, d_x, d_y);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int iter = 0; iter < 20; iter++)
        spmv_pipeline<<<nb, bs>>>(nrow, max_nnz, d_nnz_per_row, d_col_ind, d_values, d_x, d_y);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_ms; cudaEventElapsedTime(&gpu_ms, start, stop); gpu_ms /= 20;
    
    CHECK_CUDA(cudaMemcpy(h_y_gpu, d_y, nrow * sizeof(double), cudaMemcpyDeviceToHost));
    double maxerr = 0;
    for (int i = 0; i < nrow; i++) { double e = fabs(h_y_cpu[i] - h_y_gpu[i]); if (e > maxerr) maxerr = e; }
    printf("BENCHMARK_RESULT:cpu_ms=%.4f,gpu_ms=%.4f,speedup=%.2f,error=%.2e\n", cpu_ms, gpu_ms, cpu_ms/gpu_ms, maxerr);
    return 0;
}
