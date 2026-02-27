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