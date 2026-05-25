
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
#define JACOBI_BLOCK_X 32
#endif

#ifndef JACOBI_BLOCK_Y
#define JACOBI_BLOCK_Y 8
#endif

// One sweep: out = stencil(in) on interior points.
// Uses shared-memory tiling with 1-cell halo on all sides.
__global__ void jacobi2d_sweep_kernel(const double* __restrict__ in,
                                     double* __restrict__ out,
                                     int n)
{
    // Interior coordinates covered by this block start at (1,1)
    const int j = 1 + (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x; // column
    const int i = 1 + (int)blockIdx.y * (int)blockDim.y + (int)threadIdx.y; // row

    const int tx = (int)threadIdx.x;
    const int ty = (int)threadIdx.y;

    // Shared tile: (blockDim.y+2) x (blockDim.x+2)
    extern __shared__ double s[];
    const int shW = (int)blockDim.x + 2;
    const int sx  = tx + 1;
    const int sy  = ty + 1;

    // Helper lambda for shared indexing
    auto S = [&](int y, int x) -> double& { return s[y * shW + x]; };

    // Load center
    if (i < n - 1 && j < n - 1) {
        S(sy, sx) = in[i * n + j];
    }

    // Load halos (guarded by bounds)
    if (tx == 0) {
        // left halo
        if (i < n - 1 && (j - 1) >= 0) S(sy, 0) = in[i * n + (j - 1)];
    }
    if (tx == (int)blockDim.x - 1) {
        // right halo
        if (i < n - 1 && (j + 1) < n) S(sy, sx + 1) = in[i * n + (j + 1)];
    }
    if (ty == 0) {
        // top halo
        if ((i - 1) >= 0 && j < n - 1) S(0, sx) = in[(i - 1) * n + j];
    }
    if (ty == (int)blockDim.y - 1) {
        // bottom halo
        if ((i + 1) < n && j < n - 1) S(sy + 1, sx) = in[(i + 1) * n + j];
    }

    // Load corners (optional but needed for completeness when both halo directions are used)
    if (tx == 0 && ty == 0) {
        if ((i - 1) >= 0 && (j - 1) >= 0) S(0, 0) = in[(i - 1) * n + (j - 1)];
    }
    if (tx == (int)blockDim.x - 1 && ty == 0) {
        if ((i - 1) >= 0 && (j + 1) < n) S(0, sx + 1) = in[(i - 1) * n + (j + 1)];
    }
    if (tx == 0 && ty == (int)blockDim.y - 1) {
        if ((i + 1) < n && (j - 1) >= 0) S(sy + 1, 0) = in[(i + 1) * n + (j - 1)];
    }
    if (tx == (int)blockDim.x - 1 && ty == (int)blockDim.y - 1) {
        if ((i + 1) < n && (j + 1) < n) S(sy + 1, sx + 1) = in[(i + 1) * n + (j + 1)];
    }

    __syncthreads();

    // Compute only interior points (boundaries untouched)
    if (i >= 1 && i <= n - 2 && j >= 1 && j <= n - 2) {
        const double v =
            S(sy, sx) +
            S(sy, sx - 1) +
            S(sy, sx + 1) +
            S(sy - 1, sx) +
            S(sy + 1, sx);

        out[i * n + j] = 0.2 * v;
    }
}

// Host launcher: operates on device pointers A and B (row-major n*n).
// Per time step: B = stencil(A) then A = stencil(B). Final result in A.
void jacobi_gpu(int tsteps, int n, double* A, double* B)
{
    const dim3 block(JACOBI_BLOCK_X, JACOBI_BLOCK_Y);

    // Cover interior (n-2) x (n-2), offset handled in kernel by +1
    const int interiorW = (n >= 2) ? (n - 2) : 0;
    const int interiorH = (n >= 2) ? (n - 2) : 0;

    const dim3 grid(
        (interiorW + (int)block.x - 1) / (int)block.x,
        (interiorH + (int)block.y - 1) / (int)block.y
    );

    const size_t shmemBytes = (size_t)(block.x + 2) * (size_t)(block.y + 2) * sizeof(double);

    for (int t = 0; t < tsteps; ++t) {
        // Sweep 1: A -> B
        jacobi2d_sweep_kernel<<<grid, block, shmemBytes>>>(A, B, n);
        // Sweep 2: B -> A
        jacobi2d_sweep_kernel<<<grid, block, shmemBytes>>>(B, A, n);
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
