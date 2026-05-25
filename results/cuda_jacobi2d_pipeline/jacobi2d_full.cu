
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { cudaError_t e=call; if(e!=cudaSuccess){printf("CUDA_ERROR: %s\n",cudaGetErrorString(e));exit(1);} }

#define N 2000
#define TSTEPS 20

// ============ LLM Generated code (kernels + host launcher jacobi_gpu) ============
#include <cuda_runtime.h>

#ifndef JACOBI_BLOCK_X
#define JACOBI_BLOCK_X 32
#endif

#ifndef JACOBI_BLOCK_Y
#define JACOBI_BLOCK_Y 8
#endif

// One sweep: out = stencil(in) on interior points, using shared-memory tiling + 1-cell halo.
__global__ void jacobi2d_sweep_shmem(const double* __restrict__ in,
                                    double* __restrict__ out,
                                    int n)
{
    // Shared tile with halo: (blockDim.y+2) x (blockDim.x+2)
    extern __shared__ double sh[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int tileW = blockDim.x;
    const int tileH = blockDim.y;
    const int shW   = tileW + 2;

    // Global indices for interior mapping (+1 shift to skip boundary)
    const int j = blockIdx.x * tileW + tx + 1; // col
    const int i = blockIdx.y * tileH + ty + 1; // row

    // Shared indices (+1 for halo offset)
    const int sj = tx + 1;
    const int si = ty + 1;

    auto sh_at = [&](int r, int c) -> double& { return sh[r * shW + c]; };

    // Helper to safely load from global (guarded to avoid OOB)
    auto gload = [&](int gr, int gc) -> double {
        return in[gr * n + gc];
    };

    // Load center (only if within interior bounds; otherwise load something valid to avoid OOB)
    if (i <= n - 2 && j <= n - 2) {
        sh_at(si, sj) = gload(i, j);
    } else {
        // For threads outside interior, clamp to nearest valid index (keeps loads in-bounds)
        int ci = (i < 0) ? 0 : (i > n - 1 ? n - 1 : i);
        int cj = (j < 0) ? 0 : (j > n - 1 ? n - 1 : j);
        sh_at(si, sj) = gload(ci, cj);
    }

    // Halo loads: only threads on tile borders participate, with bounds checks.
    // Left halo
    if (tx == 0) {
        int gj = j - 1;
        int gi = i;
        if (gi >= 0 && gi < n && gj >= 0 && gj < n) sh_at(si, 0) = gload(gi, gj);
        else sh_at(si, 0) = sh_at(si, sj);
    }
    // Right halo
    if (tx == tileW - 1) {
        int gj = j + 1;
        int gi = i;
        if (gi >= 0 && gi < n && gj >= 0 && gj < n) sh_at(si, sj + 1) = gload(gi, gj);
        else sh_at(si, sj + 1) = sh_at(si, sj);
    }
    // Top halo
    if (ty == 0) {
        int gi = i - 1;
        int gj = j;
        if (gi >= 0 && gi < n && gj >= 0 && gj < n) sh_at(0, sj) = gload(gi, gj);
        else sh_at(0, sj) = sh_at(si, sj);
    }
    // Bottom halo
    if (ty == tileH - 1) {
        int gi = i + 1;
        int gj = j;
        if (gi >= 0 && gi < n && gj >= 0 && gj < n) sh_at(si + 1, sj) = gload(gi, gj);
        else sh_at(si + 1, sj) = sh_at(si, sj);
    }

    // Corner halos (4 threads)
    if (tx == 0 && ty == 0) {
        int gi = i - 1, gj = j - 1;
        if (gi >= 0 && gi < n && gj >= 0 && gj < n) sh_at(0, 0) = gload(gi, gj);
        else sh_at(0, 0) = sh_at(si, sj);
    }
    if (tx == tileW - 1 && ty == 0) {
        int gi = i - 1, gj = j + 1;
        if (gi >= 0 && gi < n && gj >= 0 && gj < n) sh_at(0, sj + 1) = gload(gi, gj);
        else sh_at(0, sj + 1) = sh_at(si, sj);
    }
    if (tx == 0 && ty == tileH - 1) {
        int gi = i + 1, gj = j - 1;
        if (gi >= 0 && gi < n && gj >= 0 && gj < n) sh_at(si + 1, 0) = gload(gi, gj);
        else sh_at(si + 1, 0) = sh_at(si, sj);
    }
    if (tx == tileW - 1 && ty == tileH - 1) {
        int gi = i + 1, gj = j + 1;
        if (gi >= 0 && gi < n && gj >= 0 && gj < n) sh_at(si + 1, sj + 1) = gload(gi, gj);
        else sh_at(si + 1, sj + 1) = sh_at(si, sj);
    }

    __syncthreads();

    // Compute only for interior points
    if (i <= n - 2 && j <= n - 2) {
        const double center = sh_at(si, sj);
        const double left   = sh_at(si, sj - 1);
        const double right  = sh_at(si, sj + 1);
        const double up     = sh_at(si - 1, sj);
        const double down   = sh_at(si + 1, sj);

        out[i * n + j] = 0.2 * (center + left + right + down + up);
    }
}

// Host launcher: device pointers A, B are n*n row-major on device.
// Performs tsteps of: B=stencil(A) then A=stencil(B). Final result in A.
void jacobi_gpu(int tsteps, int n, double* A, double* B)
{
    const dim3 block(JACOBI_BLOCK_X, JACOBI_BLOCK_Y);
    const int interiorW = (n >= 2) ? (n - 2) : 0;
    const int interiorH = (n >= 2) ? (n - 2) : 0;
    const dim3 grid((interiorW + block.x - 1) / block.x,
                    (interiorH + block.y - 1) / block.y);

    const size_t shmemBytes = (size_t)(block.x + 2) * (size_t)(block.y + 2) * sizeof(double);

    for (int t = 0; t < tsteps; ++t) {
        jacobi2d_sweep_shmem<<<grid, block, shmemBytes>>>(A, B, n);
        jacobi2d_sweep_shmem<<<grid, block, shmemBytes>>>(B, A, n);
    }
}

// ============ Strict serial CPU reference ============
static void jacobi_cpu(int tsteps, int n, double* A, double* B) {
    for (int t = 0; t < tsteps; t++) {
        for (int i = 1; i < n - 1; i++)
            for (int j = 1; j < n - 1; j++)
                B[i*n+j] = 0.2 * (A[i*n+j] + A[i*n+(j-1)] + A[i*n+(1+j)] + A[(1+i)*n+j] + A[(i-1)*n+j]);
        for (int i = 1; i < n - 1; i++)
            for (int j = 1; j < n - 1; j++)
                A[i*n+j] = 0.2 * (B[i*n+j] + B[i*n+(j-1)] + B[i*n+(1+j)] + B[(1+i)*n+j] + B[(i-1)*n+j]);
    }
}

static void init_arrays(int n, double* A, double* B) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            A[i*n+j] = ((double)i*(j+2) + 2) / n;
            B[i*n+j] = ((double)i*(j+3) + 3) / n;
        }
}

static double max_abs_diff(const double* a, const double* b, int n) {
    double m = 0.0;
    for (int k = 0; k < n*n; k++) { double e = fabs(a[k]-b[k]); if (e>m) m=e; }
    return m;
}

int main() {
    int n = N, tsteps = TSTEPS;
    size_t bytes = (size_t)n*n*sizeof(double);

    double* A0 = (double*)malloc(bytes);
    double* B0 = (double*)malloc(bytes);
    double* Ac = (double*)malloc(bytes);
    double* Bc = (double*)malloc(bytes);
    double* Ag = (double*)malloc(bytes);

    init_arrays(n, A0, B0);

    // CPU reference (timed, 3 runs averaged)
    double cpu_ms = 0.0;
    for (int rep = 0; rep < 3; rep++) {
        for (size_t k = 0; k < (size_t)n*n; k++) { Ac[k]=A0[k]; Bc[k]=B0[k]; }
        clock_t s = clock();
        jacobi_cpu(tsteps, n, Ac, Bc);
        clock_t e = clock();
        cpu_ms += (double)(e-s)/CLOCKS_PER_SEC*1000.0;
    }
    cpu_ms /= 3.0;

    // GPU
    double *dA, *dB;
    CHECK_CUDA(cudaMalloc(&dA, bytes));
    CHECK_CUDA(cudaMalloc(&dB, bytes));

    // warmup + correctness
    CHECK_CUDA(cudaMemcpy(dA, A0, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, B0, bytes, cudaMemcpyHostToDevice));
    jacobi_gpu(tsteps, n, dA, dB);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(Ag, dA, bytes, cudaMemcpyDeviceToHost));

    // timed (5 runs)
    cudaEvent_t st, sp; cudaEventCreate(&st); cudaEventCreate(&sp);
    cudaEventRecord(st);
    for (int rep = 0; rep < 5; rep++) {
        CHECK_CUDA(cudaMemcpy(dA, A0, bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dB, B0, bytes, cudaMemcpyHostToDevice));
        jacobi_gpu(tsteps, n, dA, dB);
    }
    cudaEventRecord(sp); cudaEventSynchronize(sp);
    float gpu_ms = 0.0f; cudaEventElapsedTime(&gpu_ms, st, sp); gpu_ms /= 5.0f;

    double err = max_abs_diff(Ac, Ag, n);

    printf("BENCHMARK_RESULT:cpu_ms=%.4f,gpu_ms=%.4f,speedup=%.2f,error=%.2e\n",
           cpu_ms, gpu_ms, cpu_ms/(double)gpu_ms, err);

    cudaFree(dA); cudaFree(dB);
    free(A0); free(B0); free(Ac); free(Bc); free(Ag);
    return 0;
}
