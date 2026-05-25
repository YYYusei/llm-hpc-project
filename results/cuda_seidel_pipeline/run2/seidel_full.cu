
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#define CHECK_CUDA(call) { cudaError_t e=call; if(e!=cudaSuccess){printf("CUDA_ERROR: %s\n",cudaGetErrorString(e));exit(1);} }
#define N 2000
#define TSTEPS 20
// ===== LLM Generated (kernel + host launcher seidel_gpu) =====
#include <cuda_runtime.h>

#ifndef SEIDEL_TILE_B
#define SEIDEL_TILE_B 16
#endif

// Blocked wavefront Gauss–Seidel:
// - One CUDA block updates one BxB interior tile.
// - Tiles are launched in wavefront order of (tile_i + tile_j) = s.
// - Inside a tile, updates proceed along anti-diagonals with __syncthreads().
// This preserves the lexicographic in-place Gauss–Seidel semantics.
__global__ void seidel2d_tile_wavefront_kernel(int n, double* __restrict__ A, int s)
{
    constexpr int B = SEIDEL_TILE_B;

    // Tile grid over interior [1..n-2]x[1..n-2]
    const int interior = n - 2;
    const int ntx = (interior + B - 1) / B;
    const int nty = (interior + B - 1) / B;

    // Enumerate tiles on this wavefront s
    const int tiles_on_s = min(ntx, s + 1) - max(0, s - (nty - 1));
    const int idx = (int)blockIdx.x;
    if (idx >= tiles_on_s) return;

    const int tx_min = max(0, s - (nty - 1));
    const int tile_x = tx_min + idx;
    const int tile_y = s - tile_x;

    if (tile_x < 0 || tile_x >= ntx || tile_y < 0 || tile_y >= nty) return;

    // Global start indices (interior coordinates)
    const int i0 = 1 + tile_y * B;
    const int j0 = 1 + tile_x * B;

    // Actual tile extents (for boundary tiles)
    const int tile_h = min(B, (n - 1) - i0); // i in [i0 .. i0+tile_h-1] <= n-2
    const int tile_w = min(B, (n - 1) - j0); // j in [j0 .. j0+tile_w-1] <= n-2

    // Shared memory tile with 1-cell halo on all sides
    __shared__ double sh[B + 2][B + 2];

    // Load (tile_h+2) x (tile_w+2) region into shared memory.
    // Halo indices map to global [i0-1 .. i0+tile_h] and [j0-1 .. j0+tile_w]
    for (int li = threadIdx.y; li < tile_h + 2; li += blockDim.y) {
        const int gi = i0 - 1 + li;
        for (int lj = threadIdx.x; lj < tile_w + 2; lj += blockDim.x) {
            const int gj = j0 - 1 + lj;
            sh[li][lj] = A[gi * n + gj];
        }
    }
    __syncthreads();

    // Intra-tile anti-diagonals over local interior cells (1..tile_h, 1..tile_w)
    // dd = (li-1) + (lj-1) ranges 0..(tile_h-1 + tile_w-1)
    const int max_dd = (tile_h - 1) + (tile_w - 1);
    for (int dd = 0; dd <= max_dd; ++dd) {
        // Map threads to candidate (li, lj) pairs on this diagonal.
        // Use a simple 2D threadblock mapping; each thread checks one (li,lj).
        const int li = 1 + (int)threadIdx.y;
        const int lj = 1 + (int)threadIdx.x;

        if (li <= tile_h && lj <= tile_w) {
            if ((li - 1) + (lj - 1) == dd) {
                const double v =
                    (sh[li - 1][lj - 1] + sh[li - 1][lj] + sh[li - 1][lj + 1] +
                     sh[li][lj - 1]     + sh[li][lj]     + sh[li][lj + 1] +
                     sh[li + 1][lj - 1] + sh[li + 1][lj] + sh[li + 1][lj + 1]) * (1.0 / 9.0);
                sh[li][lj] = v;
            }
        }
        __syncthreads();
    }

    // Store updated interior tile back to global memory
    for (int li = threadIdx.y; li < tile_h; li += blockDim.y) {
        const int gi = i0 + li;
        for (int lj = threadIdx.x; lj < tile_w; lj += blockDim.x) {
            const int gj = j0 + lj;
            A[gi * n + gj] = sh[li + 1][lj + 1];
        }
    }
}

void seidel_gpu(int tsteps, int n, double* A)
{
    if (tsteps <= 0 || n < 3 || A == nullptr) return;

    constexpr int B = SEIDEL_TILE_B;

    const int interior = n - 2;
    const int ntx = (interior + B - 1) / B;
    const int nty = (interior + B - 1) / B;
    const int num_wavefronts = ntx + nty - 1;

    // Threads: BxB so each thread can own one interior cell in the tile-diagonal step.
    dim3 block(B, B, 1);

    for (int t = 0; t < tsteps; ++t) {
        for (int s = 0; s < num_wavefronts; ++s) {
            const int tiles_on_s = min(ntx, s + 1) - max(0, s - (nty - 1));
            if (tiles_on_s <= 0) continue;

            dim3 grid((unsigned)tiles_on_s, 1, 1);
            seidel2d_tile_wavefront_kernel<<<grid, block>>>(n, A, s);
        }
        // Ensure all wavefront launches for this timestep are complete before next timestep.
        cudaDeviceSynchronize();
    }
}
// ===== Strict serial CPU reference =====
static void seidel_cpu(int tsteps, int n, double* A) {
    for (int t = 0; t <= tsteps - 1; t++)
        for (int i = 1; i <= n - 2; i++)
            for (int j = 1; j <= n - 2; j++)
                A[i*n + j] = (A[(i-1)*n + (j-1)] + A[(i-1)*n + j] + A[(i-1)*n + (j+1)]
                            + A[i*n + (j-1)]     + A[i*n + j]     + A[i*n + (j+1)]
                            + A[(i+1)*n + (j-1)] + A[(i+1)*n + j] + A[(i+1)*n + (j+1)]) / 9.0;
}
static void init_array(int n, double* A, unsigned seed) {
    srand(seed);
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++)
        A[i*n + j] = (double)((i*(j+2) + 2) % 100) / n;
}
static double max_abs_diff(const double* a, const double* b, int n) {
    double m=0.0; for (int k=0;k<n*n;k++){double e=fabs(a[k]-b[k]); if(e>m)m=e;} return m;
}
int main() {
    int n=N, tsteps=TSTEPS; size_t bytes=(size_t)n*n*sizeof(double);
    double* h_cpu=(double*)malloc(bytes);
    double* h_gpu=(double*)malloc(bytes);
    double* h_A0 =(double*)malloc(bytes);
    init_array(n, h_A0, 12345u);
    double cpu_ms=0.0;
    for (int rep=0; rep<3; rep++) {
        for (int k=0;k<n*n;k++) h_cpu[k]=h_A0[k];
        clock_t s=clock(); seidel_cpu(tsteps,n,h_cpu); clock_t e=clock();
        cpu_ms += (double)(e-s)/CLOCKS_PER_SEC*1000.0;
    }
    cpu_ms/=3.0;
    double* d_A; CHECK_CUDA(cudaMalloc(&d_A,bytes));
    CHECK_CUDA(cudaMemcpy(d_A,h_A0,bytes,cudaMemcpyHostToDevice));
    seidel_gpu(tsteps,n,d_A);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_gpu,d_A,bytes,cudaMemcpyDeviceToHost));
    cudaEvent_t st,sp; cudaEventCreate(&st); cudaEventCreate(&sp);
    cudaEventRecord(st);
    for (int rep=0; rep<5; rep++) {
        CHECK_CUDA(cudaMemcpy(d_A,h_A0,bytes,cudaMemcpyHostToDevice));
        seidel_gpu(tsteps,n,d_A);
    }
    cudaEventRecord(sp); cudaEventSynchronize(sp);
    float gpu_ms=0.0f; cudaEventElapsedTime(&gpu_ms,st,sp); gpu_ms/=5.0f;
    double err=max_abs_diff(h_cpu,h_gpu,n);
    printf("BENCHMARK_RESULT:cpu_ms=%.4f,gpu_ms=%.4f,speedup=%.2f,error=%.2e\n",
           cpu_ms,gpu_ms,cpu_ms/(double)gpu_ms,err);
    cudaFree(d_A); free(h_cpu); free(h_gpu); free(h_A0);
    return 0;
}
