
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
#define SEIDEL_TILE_B 32
#endif

// One block processes one BxB tile (interior only), in tile-wavefront order.
// Within the tile, we perform local anti-diagonal (wavefront) updates in shared memory.
__global__ void seidel2d_tile_wavefront_kernel(double* __restrict__ A, int n,
                                               int tilesPerDim, int ws)
{
    constexpr int B = SEIDEL_TILE_B;

    // Map this block to a tile (ti, tj) such that ti + tj == ws
    int ti = blockIdx.x;
    int tj = ws - ti;
    if (tj < 0 || tj >= tilesPerDim) return;

    // Global coordinates of the tile's top-left interior element
    int i0 = 1 + ti * B;
    int j0 = 1 + tj * B;

    // Shared memory for (B+2)x(B+2) including 1-cell halo
    __shared__ double sh[(B + 2) * (B + 2)];
    auto SH = [&](int si, int sj) -> double& { return sh[si * (B + 2) + sj]; };

    // Cooperative load of the tile region + halo from global memory.
    // We clamp loads to [0, n-1] to safely handle partial tiles near the boundary.
    int tid = threadIdx.x;
    int total = (B + 2) * (B + 2);
    for (int idx = tid; idx < total; idx += blockDim.x) {
        int si = idx / (B + 2);
        int sj = idx - si * (B + 2);

        int gi = i0 + (si - 1);
        int gj = j0 + (sj - 1);

        gi = (gi < 0) ? 0 : (gi >= n ? (n - 1) : gi);
        gj = (gj < 0) ? 0 : (gj >= n ? (n - 1) : gj);

        SH(si, sj) = A[gi * n + gj];
    }
    __syncthreads();

    // Local wavefront updates over the BxB interior of the tile.
    // Only update points that are truly interior globally: 1..n-2.
    // Local interior indices: li, lj in [0, B-1] correspond to shared [li+1][lj+1].
    for (int k = 0; k <= 2 * (B - 1); ++k) {
        // Each thread handles multiple (li, lj) pairs on this diagonal via linear enumeration.
        // Enumerate all B*B points; update those with li+lj==k.
        for (int p = tid; p < B * B; p += blockDim.x) {
            int li = p / B;
            int lj = p - li * B;
            if (li + lj != k) continue;

            int gi = i0 + li;
            int gj = j0 + lj;

            // Only update global interior points
            if (gi >= 1 && gi <= n - 2 && gj >= 1 && gj <= n - 2) {
                int si = li + 1;
                int sj = lj + 1;

                double sum =
                    SH(si - 1, sj - 1) + SH(si - 1, sj) + SH(si - 1, sj + 1) +
                    SH(si,     sj - 1) + SH(si,     sj) + SH(si,     sj + 1) +
                    SH(si + 1, sj - 1) + SH(si + 1, sj) + SH(si + 1, sj + 1);

                SH(si, sj) = sum * (1.0 / 9.0);
            }
        }
        __syncthreads(); // enforce local wavefront dependency order
    }

    // Store back the updated BxB interior to global memory (only true interior points).
    for (int p = tid; p < B * B; p += blockDim.x) {
        int li = p / B;
        int lj = p - li * B;

        int gi = i0 + li;
        int gj = j0 + lj;

        if (gi >= 1 && gi <= n - 2 && gj >= 1 && gj <= n - 2) {
            A[gi * n + gj] = SH(li + 1, lj + 1);
        }
    }
}

// Host launcher required by the prompt.
// A is an n*n row-major array already resident in device memory.
void seidel_gpu(int tsteps, int n, double* A)
{
    if (tsteps <= 0 || n <= 2 || A == nullptr) return;

    constexpr int B = SEIDEL_TILE_B;

    // Number of tiles covering the interior extent (1..n-2) in each dimension
    int interior = n - 2;
    int tilesPerDim = (interior + B - 1) / B;
    int maxWs = 2 * (tilesPerDim - 1);

    // Threads per block: 256 is a good default for cooperative loads and updates.
    dim3 block(256);

    for (int t = 0; t < tsteps; ++t) {
        for (int ws = 0; ws <= maxWs; ++ws) {
            // For a given ws, valid ti are [max(0, ws-(tilesPerDim-1)) .. min(tilesPerDim-1, ws)]
            int ti_min = (ws - (tilesPerDim - 1) > 0) ? (ws - (tilesPerDim - 1)) : 0;
            int ti_max = (ws < (tilesPerDim - 1)) ? ws : (tilesPerDim - 1);
            int numTilesThisWs = ti_max - ti_min + 1;
            if (numTilesThisWs <= 0) continue;

            // Launch blocks indexed by local ti' in [0..numTilesThisWs-1], map to ti = ti_min + blockIdx.x
            // We implement this by passing ws and tilesPerDim, and offsetting ti inside kernel via grid-stride:
            // simplest: launch exactly numTilesThisWs blocks and add ti_min to blockIdx.x in-kernel.
            // To avoid changing kernel signature, we incorporate ti_min by shifting ws and tilesPerDim? Not possible.
            // So we use a small wrapper: encode ti_min into ws via negative? Not allowed.
            // Instead, we launch tilesPerDim blocks and let kernel return for invalid tj; but that wastes blocks.
            // We'll do the efficient way by using a separate kernel that adds ti_min.
            // However prompt allows "any kernels it needs"; so we provide a tiny wrapper kernel.

            // Wrapper launch:
            // grid.x = numTilesThisWs, kernel computes ti = ti_min + blockIdx.x, tj = ws - ti.
            // We'll call a specialized kernel via a lambda-like pattern by using a second kernel below.
            extern __global__ void seidel2d_tile_wavefront_kernel_offset(double*, int, int, int, int);
            dim3 grid(numTilesThisWs);
            seidel2d_tile_wavefront_kernel_offset<<<grid, block>>>(A, n, tilesPerDim, ws, ti_min);
        }
    }
}

// Offset wrapper kernel (declared after seidel_gpu uses it via extern).
__global__ void seidel2d_tile_wavefront_kernel_offset(double* __restrict__ A, int n,
                                                      int tilesPerDim, int ws, int ti_min)
{
    int ti = ti_min + blockIdx.x;
    int tj = ws - ti;
    if (ti < 0 || ti >= tilesPerDim || tj < 0 || tj >= tilesPerDim) return;

    // Reuse the main kernel logic by inlining it here (cannot launch a kernel from a kernel without dynamic parallelism).
    constexpr int B = SEIDEL_TILE_B;

    int i0 = 1 + ti * B;
    int j0 = 1 + tj * B;

    __shared__ double sh[(B + 2) * (B + 2)];
    auto SH = [&](int si, int sj) -> double& { return sh[si * (B + 2) + sj]; };

    int tid = threadIdx.x;
    int total = (B + 2) * (B + 2);
    for (int idx = tid; idx < total; idx += blockDim.x) {
        int si = idx / (B + 2);
        int sj = idx - si * (B + 2);

        int gi = i0 + (si - 1);
        int gj = j0 + (sj - 1);

        gi = (gi < 0) ? 0 : (gi >= n ? (n - 1) : gi);
        gj = (gj < 0) ? 0 : (gj >= n ? (n - 1) : gj);

        SH(si, sj) = A[gi * n + gj];
    }
    __syncthreads();

    for (int k = 0; k <= 2 * (B - 1); ++k) {
        for (int p = tid; p < B * B; p += blockDim.x) {
            int li = p / B;
            int lj = p - li * B;
            if (li + lj != k) continue;

            int gi = i0 + li;
            int gj = j0 + lj;

            if (gi >= 1 && gi <= n - 2 && gj >= 1 && gj <= n - 2) {
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

    for (int p = tid; p < B * B; p += blockDim.x) {
        int li = p / B;
        int lj = p - li * B;

        int gi = i0 + li;
        int gj = j0 + lj;

        if (gi >= 1 && gi <= n - 2 && gj >= 1 && gj <= n - 2) {
            A[gi * n + gj] = SH(li + 1, lj + 1);
        }
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
