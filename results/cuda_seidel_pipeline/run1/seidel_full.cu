
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

#ifndef TILE
#define TILE 16  // interior tile size (T). Shared tile is (T+2)x(T+2).
#endif

// One kernel processes all tiles on a given tile anti-diagonal: bi + bj == tileWave.
// Each block handles one tile (bi,bj) and performs an in-tile wavefront (anti-diagonal)
// Gauss–Seidel update using shared memory and __syncthreads() between local wavefronts.
__global__ void seidel2d_tilewave_kernel(int n, double* __restrict__ A, int tileWave)
{
    // Number of interior points per dimension is (n-2). Tiles cover that interior.
    const int interior = n - 2;
    if (interior <= 0) return;

    const int numTiles = (interior + TILE - 1) / TILE;

    // Map blockIdx.x to a unique (bi,bj) such that bi+bj == tileWave.
    // Enumerate bi in [max(0, tileWave-(numTiles-1)) .. min(numTiles-1, tileWave)].
    const int bi_min = max(0, tileWave - (numTiles - 1));
    const int bi_max = min(numTiles - 1, tileWave);
    const int count  = bi_max - bi_min + 1;
    if (count <= 0) return;

    const int k = (int)blockIdx.x;
    if (k >= count) return;

    const int bi = bi_min + k;
    const int bj = tileWave - bi;

    // Global interior start indices for this tile (in full matrix coordinates).
    const int i0 = 1 + bi * TILE;
    const int j0 = 1 + bj * TILE;

    // Actual interior extents for partial tiles at the boundary.
    const int ti = min(TILE, (n - 1) - i0); // max i is n-2, so count is (n-1 - i0)
    const int tj = min(TILE, (n - 1) - j0);

    if (ti <= 0 || tj <= 0) return;

    // Shared memory tile with halo: indices [0..ti+1][0..tj+1] within allocated (TILE+2)^2.
    __shared__ double sh[(TILE + 2) * (TILE + 2)];
    auto SH = [&](int si, int sj) -> double& { return sh[si * (TILE + 2) + sj]; };

    // Load (ti+2) x (tj+2) region from global A into shared (including halo).
    // Threads cooperate over the full shared tile footprint.
    const int tid = (int)threadIdx.x;
    const int threads = (int)blockDim.x;

    const int sh_rows = ti + 2;
    const int sh_cols = tj + 2;
    const int sh_elems = sh_rows * sh_cols;

    for (int idx = tid; idx < sh_elems; idx += threads) {
        int si = idx / sh_cols; // 0..ti+1
        int sj = idx - si * sh_cols; // 0..tj+1

        int gi = i0 + (si - 1);
        int gj = j0 + (sj - 1);

        // gi,gj are guaranteed within [0..n-1] because i0>=1, j0>=1 and halo extends by 1.
        SH(si, sj) = A[gi * n + gj];
    }
    __syncthreads();

    // In-tile wavefront updates over interior shared indices [1..ti][1..tj].
    // localWave = (li + lj) where li in [0..ti-1], lj in [0..tj-1].
    // Update SH(li+1, lj+1) using 3x3 neighbors in shared.
    const int localWaves = (ti - 1) + (tj - 1); // max (li+lj)
    for (int w = 0; w <= localWaves; ++w) {
        // Each thread updates at most one point on this local anti-diagonal.
        // Map threadIdx.x to a candidate li; compute lj = w - li.
        int li = tid; // 0..(threads-1)
        if (li < ti) {
            int lj = w - li;
            if (0 <= lj && lj < tj) {
                int si = li + 1;
                int sj = lj + 1;

                double sum =
                    SH(si - 1, sj - 1) + SH(si - 1, sj) + SH(si - 1, sj + 1) +
                    SH(si,     sj - 1) + SH(si,     sj) + SH(si,     sj + 1) +
                    SH(si + 1, sj - 1) + SH(si + 1, sj) + SH(si + 1, sj + 1);

                SH(si, sj) = sum * (1.0 / 9.0);
            }
        }
        __syncthreads();
    }

    // Write updated interior back to global A.
    const int interior_elems = ti * tj;
    for (int idx = tid; idx < interior_elems; idx += threads) {
        int li = idx / tj;      // 0..ti-1
        int lj = idx - li * tj; // 0..tj-1
        int gi = i0 + li;
        int gj = j0 + lj;
        A[gi * n + gj] = SH(li + 1, lj + 1);
    }
}

// Host launcher: performs tsteps Gauss–Seidel sweeps in-place on device array A.
// Uses global tile-wavefront ordering (one kernel launch per tileWave per time step).
void seidel_gpu(int tsteps, int n, double* A)
{
    if (tsteps <= 0 || n <= 2 || A == nullptr) return;

    const int interior = n - 2;
    const int numTiles = (interior + TILE - 1) / TILE;
    const int maxTileWave = (numTiles - 1) + (numTiles - 1);

    // Threads per block: choose at least TILE threads so each localWave can be covered.
    // (Correctness does not require full coverage in one step because threads stride,
    // but using >= TILE is important for good coverage of li dimension.)
    const int threads = 256; // must be >= TILE; 256 is a reasonable default.

    for (int t = 0; t < tsteps; ++t) {
        for (int tileWave = 0; tileWave <= maxTileWave; ++tileWave) {
            const int bi_min = max(0, tileWave - (numTiles - 1));
            const int bi_max = min(numTiles - 1, tileWave);
            const int blocks = (bi_max >= bi_min) ? (bi_max - bi_min + 1) : 0;
            if (blocks > 0) {
                seidel2d_tilewave_kernel<<<blocks, threads>>>(n, A, tileWave);
            }
        }
    }

    // Ensure completion before returning (caller expects results ready on device).
    cudaDeviceSynchronize();
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
