
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#define CHECK_CUDA(call) { cudaError_t e=call; if(e!=cudaSuccess){printf("CUDA_ERROR: %s\n",cudaGetErrorString(e));exit(1);} }
#define N 2000
#define TSTEPS 20
// ===== LLM Generated (kernels + host launcher jacobi_gpu) =====
#include <cuda_runtime.h>

#ifndef JACOBI_BLOCK_X
#define JACOBI_BLOCK_X 16
#endif

#ifndef JACOBI_BLOCK_Y
#define JACOBI_BLOCK_Y 16
#endif

// One Jacobi sweep: out = stencil(in) on interior [1..n-2]x[1..n-2].
// Shared-memory tiling with 1-cell halo on all sides.
__global__ void jacobi_sweep_tiled(const double* __restrict__ in,
                                  double* __restrict__ out,
                                  int n)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int bdx = blockDim.x;
    const int bdy = blockDim.y;

    // Global indices for interior, offset by +1 to skip boundary.
    const int j = blockIdx.x * bdx + tx + 1;
    const int i = blockIdx.y * bdy + ty + 1;

    // Shared tile dimensions include halo.
    const int sh_w = bdx + 2;
    const int sh_h = bdy + 2;

    extern __shared__ double sh[]; // size = (bdx+2)*(bdy+2)
    auto sh_at = [&](int y, int x) -> double& { return sh[y * sh_w + x]; };

    // Cooperative loading of the entire shared tile (including halo)
    // using a linearized loop to avoid branch-heavy halo code.
    const int tile_elems = sh_w * sh_h;
    const int threads = bdx * bdy;
    const int tid = ty * bdx + tx;

    // Top-left global coordinate of the shared tile (including halo).
    const int base_i = blockIdx.y * bdy; // corresponds to global i = base_i + sy, where sy in [0..bdy+1]
    const int base_j = blockIdx.x * bdx; // corresponds to global j = base_j + sx, where sx in [0..bdx+1]

    for (int idx = tid; idx < tile_elems; idx += threads) {
        const int sy = idx / sh_w; // 0..bdy+1
        const int sx = idx - sy * sh_w; // 0..bdx+1

        const int gi = base_i + sy; // 0..n-1 (may exceed at edges)
        const int gj = base_j + sx; // 0..n-1 (may exceed at edges)

        // Clamp to valid range to avoid OOB loads for partial blocks at domain edges.
        // This does not affect correctness because we never compute boundary outputs,
        // and interior blocks are fully in-range.
        const int cgi = (gi < 0) ? 0 : (gi >= n ? (n - 1) : gi);
        const int cgj = (gj < 0) ? 0 : (gj >= n ? (n - 1) : gj);

        sh_at(sy, sx) = in[cgi * n + cgj];
    }

    __syncthreads();

    // Compute only interior points.
    if (i < n - 1 && j < n - 1) {
        const int sy = ty + 1;
        const int sx = tx + 1;

        const double center = sh_at(sy, sx);
        const double left   = sh_at(sy, sx - 1);
        const double right  = sh_at(sy, sx + 1);
        const double down   = sh_at(sy + 1, sx);
        const double up     = sh_at(sy - 1, sx);

        out[i * n + j] = 0.2 * (center + left + right + down + up);
    }
}

// Host launcher required by prompt.
// A and B are device pointers to row-major n*n doubles.
// Performs: for each t: B=stencil(A) then A=stencil(B). Final result in A.
void jacobi_gpu(int tsteps, int n, double* A, double* B)
{
    dim3 block(JACOBI_BLOCK_X, JACOBI_BLOCK_Y);
    dim3 grid((n - 2 + block.x - 1) / block.x,
              (n - 2 + block.y - 1) / block.y);

    const size_t shmem_bytes = (size_t)(block.x + 2) * (size_t)(block.y + 2) * sizeof(double);

    for (int t = 0; t < tsteps; ++t) {
        jacobi_sweep_tiled<<<grid, block, shmem_bytes>>>(A, B, n);
        jacobi_sweep_tiled<<<grid, block, shmem_bytes>>>(B, A, n);
    }
}
// ===== Strict serial CPU reference =====
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
    for (int i=0;i<n;i++) for (int j=0;j<n;j++) {
        A[i*n+j]=((double)i*(j+2)+2)/n; B[i*n+j]=((double)i*(j+3)+3)/n;
    }
}
static double max_abs_diff(const double* a, const double* b, int n) {
    double m=0.0; for (int k=0;k<n*n;k++){double e=fabs(a[k]-b[k]); if(e>m)m=e;} return m;
}
int main() {
    int n=N, tsteps=TSTEPS; size_t bytes=(size_t)n*n*sizeof(double);
    double* A0=(double*)malloc(bytes);
    double* B0=(double*)malloc(bytes);
    double* Ac=(double*)malloc(bytes);
    double* Bc=(double*)malloc(bytes);
    double* Ag=(double*)malloc(bytes);
    init_arrays(n, A0, B0);
    double cpu_ms=0.0;
    for (int rep=0; rep<3; rep++) {
        for (size_t k=0;k<(size_t)n*n;k++){ Ac[k]=A0[k]; Bc[k]=B0[k]; }
        clock_t s=clock(); jacobi_cpu(tsteps,n,Ac,Bc); clock_t e=clock();
        cpu_ms += (double)(e-s)/CLOCKS_PER_SEC*1000.0;
    }
    cpu_ms/=3.0;
    double *dA,*dB; CHECK_CUDA(cudaMalloc(&dA,bytes)); CHECK_CUDA(cudaMalloc(&dB,bytes));
    CHECK_CUDA(cudaMemcpy(dA,A0,bytes,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB,B0,bytes,cudaMemcpyHostToDevice));
    jacobi_gpu(tsteps,n,dA,dB);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(Ag,dA,bytes,cudaMemcpyDeviceToHost));
    cudaEvent_t st,sp; cudaEventCreate(&st); cudaEventCreate(&sp);
    cudaEventRecord(st);
    for (int rep=0; rep<5; rep++) {
        CHECK_CUDA(cudaMemcpy(dA,A0,bytes,cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dB,B0,bytes,cudaMemcpyHostToDevice));
        jacobi_gpu(tsteps,n,dA,dB);
    }
    cudaEventRecord(sp); cudaEventSynchronize(sp);
    float gpu_ms=0.0f; cudaEventElapsedTime(&gpu_ms,st,sp); gpu_ms/=5.0f;
    double err=max_abs_diff(Ac,Ag,n);
    printf("BENCHMARK_RESULT:cpu_ms=%.4f,gpu_ms=%.4f,speedup=%.2f,error=%.2e\n",
           cpu_ms,gpu_ms,cpu_ms/(double)gpu_ms,err);
    cudaFree(dA); cudaFree(dB);
    free(A0); free(B0); free(Ac); free(Bc); free(Ag);
    return 0;
}
