
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

#ifndef JACOBI_BX
#define JACOBI_BX 32
#endif

#ifndef JACOBI_BY
#define JACOBI_BY 8
#endif

// One sweep: out = 0.2 * (in(center) + in(left) + in(right) + in(down) + in(up))
// Uses 2D shared-memory tiling with 1-cell halo.
__global__ void jacobi2d_sweep_shared(const double* __restrict__ in,
                                      double* __restrict__ out,
                                      int n)
{
    // Global indices (i=row=y, j=col=x)
    const int j = (int)blockIdx.x * JACOBI_BX + (int)threadIdx.x;
    const int i = (int)blockIdx.y * JACOBI_BY + (int)threadIdx.y;

    // Shared tile with halo: [BY+2][BX+2]
    __shared__ double tile[JACOBI_BY + 2][JACOBI_BX + 2];

    const int tx = (int)threadIdx.x;
    const int ty = (int)threadIdx.y;

    // Load center (or 0 if OOB; computation is guarded anyway)
    if (i < n && j < n) {
        tile[ty + 1][tx + 1] = in[i * n + j];
    } else {
        tile[ty + 1][tx + 1] = 0.0;
    }

    // Load halos (only threads on block edges participate)
    // Left halo
    if (tx == 0) {
        if (i < n && (j - 1) >= 0) tile[ty + 1][0] = in[i * n + (j - 1)];
        else tile[ty + 1][0] = 0.0;
    }
    // Right halo
    if (tx == JACOBI_BX - 1) {
        if (i < n && (j + 1) < n) tile[ty + 1][JACOBI_BX + 1] = in[i * n + (j + 1)];
        else tile[ty + 1][JACOBI_BX + 1] = 0.0;
    }
    // Top halo (i-1)
    if (ty == 0) {
        if ((i - 1) >= 0 && j < n) tile[0][tx + 1] = in[(i - 1) * n + j];
        else tile[0][tx + 1] = 0.0;
    }
    // Bottom halo (i+1)
    if (ty == JACOBI_BY - 1) {
        if ((i + 1) < n && j < n) tile[JACOBI_BY + 1][tx + 1] = in[(i + 1) * n + j];
        else tile[JACOBI_BY + 1][tx + 1] = 0.0;
    }

    // Corner halos (4 threads)
    if (tx == 0 && ty == 0) {
        if ((i - 1) >= 0 && (j - 1) >= 0) tile[0][0] = in[(i - 1) * n + (j - 1)];
        else tile[0][0] = 0.0;
    }
    if (tx == JACOBI_BX - 1 && ty == 0) {
        if ((i - 1) >= 0 && (j + 1) < n) tile[0][JACOBI_BX + 1] = in[(i - 1) * n + (j + 1)];
        else tile[0][JACOBI_BX + 1] = 0.0;
    }
    if (tx == 0 && ty == JACOBI_BY - 1) {
        if ((i + 1) < n && (j - 1) >= 0) tile[JACOBI_BY + 1][0] = in[(i + 1) * n + (j - 1)];
        else tile[JACOBI_BY + 1][0] = 0.0;
    }
    if (tx == JACOBI_BX - 1 && ty == JACOBI_BY - 1) {
        if ((i + 1) < n && (j + 1) < n) tile[JACOBI_BX ? (JACOBI_BY + 1) : 0][JACOBI_BX + 1] = in[(i + 1) * n + (j + 1)];
        else tile[JACOBI_BY + 1][JACOBI_BX + 1] = 0.0;
    }

    __syncthreads();

    // Compute only interior points; boundaries are left untouched (as in reference loops).
    if (i > 0 && i < (n - 1) && j > 0 && j < (n - 1)) {
        const double center = tile[ty + 1][tx + 1];
        const double left   = tile[ty + 1][tx + 0];
        const double right  = tile[ty + 1][tx + 2];
        const double up     = tile[ty + 0][tx + 1];
        const double down   = tile[ty + 2][tx + 1];
        out[i * n + j] = 0.2 * (center + left + right + down + up);
    }
}

// Host launcher: operates on DEVICE pointers A and B (row-major n*n).
// Per timestep: B = stencil(A) then A = stencil(B). Final result in A.
extern "C" void jacobi_gpu(int tsteps, int n, double* A, double* B)
{
    const dim3 block(JACOBI_BX, JACOBI_BY);
    const dim3 grid((n + JACOBI_BX - 1) / JACOBI_BX,
                    (n + JACOBI_BY - 1) / JACOBI_BY);

    for (int t = 0; t < tsteps; ++t) {
        jacobi2d_sweep_shared<<<grid, block>>>(A, B, n);
        jacobi2d_sweep_shared<<<grid, block>>>(B, A, n);
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
