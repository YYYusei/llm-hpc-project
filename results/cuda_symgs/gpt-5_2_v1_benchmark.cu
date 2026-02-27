
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { cudaError_t e=call; if(e!=cudaSuccess){printf("CUDA_ERROR: %s\n",cudaGetErrorString(e));exit(1);} }

// ============ LLM Generated SYMGS Kernel(s) ============
// SYMGS CUDA implementation with dependency handling via graph coloring.
//
// Why coloring?
// Gauss-Seidel has true loop-carried dependencies: in the forward sweep, row i
// uses updated x from "earlier" rows; in the backward sweep it uses updated x
// from "later" rows. Directly mapping one row per thread breaks correctness.
// Graph coloring partitions rows into "colors" such that no two rows of the same
// color depend on each other (no edge between them in the matrix graph).
// Then we can process colors sequentially (global synchronization between colors),
// while processing all rows within a color in parallel.
//
// Assumptions / inputs for this kernel code:
// - Matrix stored in ELL/CSR-like fixed-stride layout:
//     values[i*max_nnz + j], col_ind[i*max_nnz + j], nnz_per_row[i]
// - diag[i] is the diagonal value for row i (pre-extracted).
// - Coloring provided by the host (or a preprocessing step):
//     num_colors
//     color_ptr[c]..color_ptr[c+1]-1 indexes into color_rows[]
//     color_rows[k] gives the row index for that position.
// - For backward sweep we process colors in reverse order. This is correct for
//   symmetric GS when coloring is built on the (undirected) sparsity graph.
//
// Note: This is "exact" colored Gauss-Seidel (not approximate Jacobi-like).
// It requires launching one kernel per color (or using cooperative groups for
// grid-wide sync). Here we use one kernel launch per color for simplicity/correctness.

extern "C" __global__
void symgs_color_sweep_kernel(
    int nrow,
    int max_nnz,
    const int* __restrict__ nnz_per_row,
    const int* __restrict__ col_ind,
    const double* __restrict__ values,
    const double* __restrict__ diag,
    const double* __restrict__ r,
    double* __restrict__ x,
    const int* __restrict__ color_rows, // rows belonging to the active color
    int color_row_count               // number of rows in the active color
) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= color_row_count) return;

  int i = color_rows[tid];

  // Standard GS row update:
  // sum = r[i] - sum_{j in row i} A(i,j)*x[j] + x[i]*A(i,i)
  // x[i] = sum / A(i,i)
  double sum = r[i];
  int row_off = i * max_nnz;
  int nnz = nnz_per_row[i];

  // Subtract full row contribution using current x (which already contains
  // updated values from previously processed colors in this sweep).
  #pragma unroll 1
  for (int j = 0; j < nnz; j++) {
    int c = col_ind[row_off + j];
    double a = values[row_off + j];
    sum -= a * x[c];
  }

  double d = diag[i];
  sum += x[i] * d;     // add back diagonal term (since it was subtracted above)
  x[i] = sum / d;
}

// Host-callable entry point (device code only requested, so this is a __global__
// "driver" kernel is NOT appropriate). Instead, provide a device-side wrapper
// that the host will implement by launching symgs_color_sweep_kernel per color.
//
// The required function name is provided as a CUDA kernel that performs one
// *entire* SYMGS using cooperative groups grid synchronization IF the user
// launches it as a cooperative kernel. If cooperative launch is not used,
// prefer launching symgs_color_sweep_kernel per color from the host.
//
// This kernel expects:
// - color_ptr size = num_colors+1
// - color_rows contains all rows grouped by color
// It processes forward colors 0..num_colors-1 then backward num_colors-1..0.
// Between colors it uses grid-wide sync (requires cooperative launch).
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

extern "C" __global__
void symgs_gpu_gpt_5_2(
    int nrow,
    int max_nnz,
    const int* __restrict__ nnz_per_row,
    const int* __restrict__ col_ind,
    const double* __restrict__ values,
    const double* __restrict__ diag,
    const double* __restrict__ r,
    double* __restrict__ x,
    int num_colors,
    const int* __restrict__ color_ptr,   // length num_colors+1
    const int* __restrict__ color_rows   // length nrow
) {
  // Dependency strategy: multi-coloring + sequential color order.
  // All rows in a color are independent -> parallel.
  // Colors are processed sequentially -> correctness.
  cg::grid_group grid = cg::this_grid();

  // Forward sweep: colors increasing
  for (int c = 0; c < num_colors; c++) {
    int begin = color_ptr[c];
    int end   = color_ptr[c + 1];
    int count = end - begin;

    // Parallel over rows in this color
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < count;
         idx += gridDim.x * blockDim.x) {

      int i = color_rows[begin + idx];

      double sum = r[i];
      int row_off = i * max_nnz;
      int nnz = nnz_per_row[i];

      #pragma unroll 1
      for (int j = 0; j < nnz; j++) {
        int col = col_ind[row_off + j];
        double a = values[row_off + j];
        sum -= a * x[col];
      }

      double d = diag[i];
      sum += x[i] * d;
      x[i] = sum / d;
    }

    // Global barrier between colors (requires cooperative launch)
    grid.sync();
  }

  // Backward sweep: colors decreasing
  for (int c = num_colors - 1; c >= 0; c--) {
    int begin = color_ptr[c];
    int end   = color_ptr[c + 1];
    int count = end - begin;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < count;
         idx += gridDim.x * blockDim.x) {

      int i = color_rows[begin + idx];

      double sum = r[i];
      int row_off = i * max_nnz;
      int nnz = nnz_per_row[i];

      #pragma unroll 1
      for (int j = 0; j < nnz; j++) {
        int col = col_ind[row_off + j];
        double a = values[row_off + j];
        sum -= a * x[col];
      }

      double d = diag[i];
      sum += x[i] * d;
      x[i] = sum / d;
    }

    grid.sync();
  }
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
    symgs_gpu_gpt_5_2(nrow, max_nnz, d_nnz_per_row, d_col_ind, d_values, d_diag, d_r, d_x);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int iter = 0; iter < 5; iter++) {
        CHECK_CUDA(cudaMemset(d_x, 0, nrow * sizeof(double)));
        symgs_gpu_gpt_5_2(nrow, max_nnz, d_nnz_per_row, d_col_ind, d_values, d_diag, d_r, d_x);
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
