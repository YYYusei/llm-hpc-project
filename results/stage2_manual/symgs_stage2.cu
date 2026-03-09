
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#define CHECK_CUDA(c) {cudaError_t e=c; if(e!=cudaSuccess){printf("CUDA_ERROR: %s\n",cudaGetErrorString(e));exit(1);}}


#define WARP_SZ 32

__global__ void symgs_kernel_mc_warp(int nrow, int max_nnz, int target_color,
    const int*    __restrict__ row_colors,
    const int*    __restrict__ col_ell,
    const double* __restrict__ val_ell,
    const double* __restrict__ invDiag,
    const double* __restrict__ r,
    double* x)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SZ;
    int lane    = threadIdx.x % WARP_SZ;
    int i       = warp_id;
    if (i >= nrow || __ldg(&row_colors[i]) != target_color) return;

    double sum = (lane == 0) ? __ldg(&r[i]) : 0.0;
    for (int j = lane; j < max_nnz; j += WARP_SZ) {
        int    col = __ldg(&col_ell[j * nrow + i]);
        double val = __ldg(&val_ell[j * nrow + i]);
        double xj  = (col != i) ? x[col] : 0.0;
        sum -= val * xj;
    }
    #pragma unroll
    for (int offset = WARP_SZ/2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) x[i] = sum * __ldg(&invDiag[i]);
}

void symgs_stage2_graph(int nrow, int max_nnz, int num_colors,
    const int* drow_colors, const int* dcol_ell, const double* dval_ell,
    const double* dinvDiag, const double* dr, double* dx)
{
    int bs=256, nb=((nrow*WARP_SZ)+bs-1)/bs;
    /* 专用 stream: cudaStreamDefault 不支持 Graph Capture */
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaGraph_t graph; cudaGraphExec_t graphExec;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
    for (int c=0; c<num_colors; c++)
        symgs_kernel_mc_warp<<<nb,bs,0,stream>>>(nrow,max_nnz,c,drow_colors,dcol_ell,dval_ell,dinvDiag,dr,dx);
    for (int c=num_colors-1; c>=0; c--)
        symgs_kernel_mc_warp<<<nb,bs,0,stream>>>(nrow,max_nnz,c,drow_colors,dcol_ell,dval_ell,dinvDiag,dr,dx);
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);
    cudaGraphExecDestroy(graphExec); cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
}


void symgs_cpu(int nrow, int max_nnz,
    const int* col_row, const double* val_row, const double* diag,
    const double* r, double* x)
{
    for (int i=0; i<nrow; i++) {
        double sum=r[i];
        for (int j=0; j<max_nnz; j++) {
            int col=col_row[i*max_nnz+j];
            if(col!=i) sum -= val_row[i*max_nnz+j]*x[col];
        }
        x[i]=sum/diag[i];
    }
    for (int i=nrow-1; i>=0; i--) {
        double sum=r[i];
        for (int j=0; j<max_nnz; j++) {
            int col=col_row[i*max_nnz+j];
            if(col!=i) sum -= val_row[i*max_nnz+j]*x[col];
        }
        x[i]=sum/diag[i];
    }
}

void compute_colors(int nrow, int* row_colors, int num_colors) {
    int nx=50, ny=50, nz=nrow/(50*50); if(nz<1)nz=1;
    for (int i=0; i<nrow; i++) {
        int iz=i/(nx*ny), iy=(i%(nx*ny))/nx, ix=i%nx;
        row_colors[i]=(ix+iy+iz)%num_colors;
    }
}

int main() {
    int nrow=50000, max_nnz=27, num_colors=8;
    int    *hcol_row = (int*)   malloc(nrow*max_nnz*sizeof(int));
    double *hval_row = (double*)malloc(nrow*max_nnz*sizeof(double));
    int    *hcol_ell = (int*)   malloc(max_nnz*nrow*sizeof(int));
    double *hval_ell = (double*)malloc(max_nnz*nrow*sizeof(double));
    double *hdiag    = (double*)malloc(nrow*sizeof(double));
    double *hinvDiag = (double*)malloc(nrow*sizeof(double));
    double *hr       = (double*)malloc(nrow*sizeof(double));
    double *hx_cpu   = (double*)malloc(nrow*sizeof(double));
    double *hx_gpu   = (double*)malloc(nrow*sizeof(double));
    int    *hcolors  = (int*)   malloc(nrow*sizeof(int));

    compute_colors(nrow,hcolors,num_colors);
    srand(12345);
    for (int i=0; i<nrow; i++) {
        hdiag[i]=26.0; hinvDiag[i]=1.0/26.0;
        hr[i]=(double)rand()/RAND_MAX;
        hx_cpu[i]=0.0; hx_gpu[i]=0.0;
        for (int j=0; j<max_nnz; j++) {
            int col; double val;
            if(j==13){col=i;val=26.0;}
            else{
                col=i+(j-13)*100+(rand()%10-5);
                if(col<0)col=0; if(col>=nrow)col=nrow-1;
                val=-1.0;
            }
            hcol_row[i*max_nnz+j]=col; hval_row[i*max_nnz+j]=val;
            hcol_ell[j*nrow+i]=col;    hval_ell[j*nrow+i]=val;
        }
    }

    for(int i=0;i<nrow;i++) hx_cpu[i]=0.0;
    symgs_cpu(nrow,max_nnz,hcol_row,hval_row,hdiag,hr,hx_cpu); /* warmup */
    struct timespec ts0,ts1;
    clock_gettime(CLOCK_MONOTONIC,&ts0);
    for (int it=0; it<5; it++) {
        for(int i=0;i<nrow;i++) hx_cpu[i]=0.0;
        symgs_cpu(nrow,max_nnz,hcol_row,hval_row,hdiag,hr,hx_cpu);
    }
    clock_gettime(CLOCK_MONOTONIC,&ts1);
    double cpu_ms=((ts1.tv_sec-ts0.tv_sec)*1e3+(ts1.tv_nsec-ts0.tv_nsec)*1e-6)/5.0;

    int    *dcol_ell,*drow_colors;
    double *dval_ell,*dinvDiag,*dr,*dx;
    CHECK_CUDA(cudaMalloc(&dcol_ell,   max_nnz*nrow*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dval_ell,   max_nnz*nrow*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dinvDiag,   nrow*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dr,         nrow*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dx,         nrow*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&drow_colors,nrow*sizeof(int)));
    CHECK_CUDA(cudaMemcpy(dcol_ell,   hcol_ell,   max_nnz*nrow*sizeof(int),   cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dval_ell,   hval_ell,   max_nnz*nrow*sizeof(double),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dinvDiag,   hinvDiag,   nrow*sizeof(double),         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dr,         hr,         nrow*sizeof(double),         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(drow_colors,hcolors,    nrow*sizeof(int),            cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemset(dx,0,nrow*sizeof(double)));
    /* warmup: build & run graph once */
    symgs_stage2_graph(nrow,max_nnz,num_colors,drow_colors,dcol_ell,dval_ell,dinvDiag,dr,dx);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start,stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int it=0; it<10; it++) {
        CHECK_CUDA(cudaMemset(dx,0,nrow*sizeof(double)));
        symgs_stage2_graph(nrow,max_nnz,num_colors,drow_colors,dcol_ell,dval_ell,dinvDiag,dr,dx);
    }
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float gpu_ms; cudaEventElapsedTime(&gpu_ms,start,stop); gpu_ms/=10;

    CHECK_CUDA(cudaMemcpy(hx_gpu,dx,nrow*sizeof(double),cudaMemcpyDeviceToHost));
    double maxerr=0;
    for (int i=0; i<nrow; i++) { double e=fabs(hx_cpu[i]-hx_gpu[i]); if(e>maxerr)maxerr=e; }
    printf("BENCHMARK_RESULT:cpu_ms=%.4f,gpu_ms=%.4f,speedup=%.2f,error=%.2e\n",
           cpu_ms,gpu_ms,cpu_ms/gpu_ms,maxerr);
    printf("NOTE: multi-color GS != strict GS, error reflects algorithm difference (expected)\n");
    return 0;
}
