
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#define CHECK_CUDA(c) {cudaError_t e=c; if(e!=cudaSuccess){printf("CUDA_ERROR: %s\n",cudaGetErrorString(e));exit(1);}}


#define WARP_SIZE 32

/* warp-per-row 必须配行主序: val[row*max_nnz+j]
   同一 warp 的 lane 0..31 访问 j=0..31 → 连续地址 → 合并访问
   (列主序 val[j*nrow+row] 配 warp-per-row 步长=nrow*8字节，完全不合并) */
__global__ void spmv_stage2(int nrow, int max_nnz,
    const int*    __restrict__ col_rm,   /* 行主序 [nrow * max_nnz] */
    const double* __restrict__ val_rm,
    const double* __restrict__ x,
    double*       __restrict__ y)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane    = threadIdx.x % WARP_SIZE;
    int row     = warp_id;
    if (row >= nrow) return;

    double sum = 0.0;
    /* 行主序: row*max_nnz+lane, row*max_nnz+lane+32, ... → 合并 */
    for (int j = lane; j < max_nnz; j += WARP_SIZE) {
        int    col = __ldg(&col_rm[row * max_nnz + j]);
        double val = __ldg(&val_rm[row * max_nnz + j]);
        sum = fma(val, __ldg(&x[col]), sum);
    }
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) y[row] = sum;
}


void spmv_cpu(int nrow, int max_nnz,
    const int* col_row, const double* val_row, const double* x, double* y)
{
    for (int i=0; i<nrow; i++) {
        double s=0;
        for (int j=0; j<max_nnz; j++)
            s += val_row[i*max_nnz+j] * x[col_row[i*max_nnz+j]];
        y[i]=s;
    }
}

int main() {
    int nrow=500000, max_nnz=27;
    int    *hcol_row = (int*)   malloc(nrow*max_nnz*sizeof(int));
    double *hval_row = (double*)malloc(nrow*max_nnz*sizeof(double));
    double *hx       = (double*)malloc(nrow*sizeof(double));
    double *hy_cpu   = (double*)malloc(nrow*sizeof(double));
    double *hy_gpu   = (double*)malloc(nrow*sizeof(double));

    srand(12345);
    for (int i=0; i<nrow; i++) {
        hx[i]=(double)rand()/RAND_MAX;
        for (int j=0; j<max_nnz; j++) {
            int col=(i+j-13+nrow)%nrow;
            double val=(j==13)?26.0:-1.0;
            hcol_row[i*max_nnz+j]=col;
            hval_row[i*max_nnz+j]=val;
        }
    }

    /* warmup */
    spmv_cpu(nrow,max_nnz,hcol_row,hval_row,hx,hy_cpu);
    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    for (int it=0; it<20; it++) spmv_cpu(nrow,max_nnz,hcol_row,hval_row,hx,hy_cpu);
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    double cpu_ms = ((ts1.tv_sec-ts0.tv_sec)*1e3 + (ts1.tv_nsec-ts0.tv_nsec)*1e-6) / 20.0;

    /* 行主序直接上传，GPU 和 CPU 用同一数组 */
    int    *dcol_rm; double *dval_rm,*dx,*dy;
    CHECK_CUDA(cudaMalloc(&dcol_rm,nrow*max_nnz*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dval_rm,nrow*max_nnz*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dx,     nrow*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dy,     nrow*sizeof(double)));
    CHECK_CUDA(cudaMemcpy(dcol_rm,hcol_row,nrow*max_nnz*sizeof(int),   cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dval_rm,hval_row,nrow*max_nnz*sizeof(double),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dx,hx,nrow*sizeof(double),cudaMemcpyHostToDevice));

    int bs=256, nb=((nrow*32)+bs-1)/bs;
    spmv_stage2<<<nb,bs>>>(nrow,max_nnz,dcol_rm,dval_rm,dx,dy);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start,stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int it=0; it<20; it++)
        spmv_stage2<<<nb,bs>>>(nrow,max_nnz,dcol_rm,dval_rm,dx,dy);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float gpu_ms; cudaEventElapsedTime(&gpu_ms,start,stop); gpu_ms/=20;

    CHECK_CUDA(cudaMemcpy(hy_gpu,dy,nrow*sizeof(double),cudaMemcpyDeviceToHost));
    double maxerr=0;
    for (int i=0; i<nrow; i++) { double e=fabs(hy_cpu[i]-hy_gpu[i]); if(e>maxerr)maxerr=e; }
    printf("BENCHMARK_RESULT:cpu_ms=%.4f,gpu_ms=%.4f,speedup=%.2f,error=%.2e\n",
           cpu_ms,gpu_ms,cpu_ms/gpu_ms,maxerr);
    return 0;
}
