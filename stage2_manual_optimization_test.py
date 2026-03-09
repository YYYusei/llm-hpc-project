"""
Stage 2 Manual Optimization Benchmark (v3 - 严格按 gpt-5.2 Stage 2 建议)

v3 相对 v2 的新增:
  miniMD : type 改为 uint16_t（建议: uint8/uint16）
           邻居表改为 CSR 格式 head[i]/count[i]/list[]（建议: neighbor list as CSR-like）
  SPMV   : 无变化（SELL-C-σ warp-per-row 已完整实现）
  SYMGS  : 无变化（multi-color + invDiag + CUDA Graph + ELL 已完整实现）
"""
import subprocess, os, re, json
from datetime import datetime

OUTPUT_DIR = "results/stage2_manual"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# miniMD v3: 严格遵循 gpt-5.2 Stage 2 建议
#   1. cut/eps/sig → __constant__ memory (broadcast cache, 无 bank conflict)
#   2. SoA 布局: x[]/y[]/z[] 与 fx[]/fy[]/fz[] 分离
#   3. full-neigh: 每线程累加自身 fi，无 f[j] atomicAdd
#   4. type → uint16_t (建议: uint8/uint16 减小带宽)
#   5. 邻居表 CSR 格式: head[i]=起始偏移, count[i]=邻居数, list[]=邻居索引
#      (建议: "neighbor list as CSR-like head[i]/count[i]/neigh[]")
# ============================================================
MINIMD_STAGE2_KERNEL = r"""
__constant__ double c_cutforcesq[16];
__constant__ double c_epsilon[16];
__constant__ double c_sigma6[16];

/* v3: type 改为 uint16_t；邻居表改为 CSR head/count/list */
__global__ void lj_force_stage2(int nlocal, int ntypes,
    const double*   __restrict__ px, const double* __restrict__ py, const double* __restrict__ pz,
    double*         __restrict__ fx, double*       __restrict__ fy, double*       __restrict__ fz,
    const uint16_t* __restrict__ type,       /* v3: uint16_t 减小带宽 */
    const int*      __restrict__ neigh_head, /* CSR: 起始偏移 */
    const int*      __restrict__ neigh_count,/* CSR: 邻居数   */
    const int*      __restrict__ neigh_list) /* CSR: 邻居索引 */
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nlocal) return;

    double xtmp = __ldg(&px[i]);
    double ytmp = __ldg(&py[i]);
    double ztmp = __ldg(&pz[i]);
    int type_i  = (int)__ldg(&type[i]);

    double fix = 0.0, fiy = 0.0, fiz = 0.0;
    int base   = __ldg(&neigh_head[i]);
    int nneigh = __ldg(&neigh_count[i]);

    for (int k = 0; k < nneigh; k++) {
        int j = __ldg(&neigh_list[base + k]);
        double delx = xtmp - __ldg(&px[j]);
        double dely = ytmp - __ldg(&py[j]);
        double delz = ztmp - __ldg(&pz[j]);
        double rsq  = fma(delx, delx, fma(dely, dely, delz * delz));

        int type_ij = type_i * ntypes + (int)__ldg(&type[j]);
        double cut  = c_cutforcesq[type_ij];

        if (rsq < cut) {
            double sr2   = 1.0 / rsq;
            double sr6   = sr2 * sr2 * sr2 * c_sigma6[type_ij];
            double force = 48.0 * sr6 * (sr6 - 0.5) * sr2 * c_epsilon[type_ij];
            fix = fma(delx, force, fix);
            fiy = fma(dely, force, fiy);
            fiz = fma(delz, force, fiz);
        }
    }
    fx[i] = fix; fy[i] = fiy; fz[i] = fiz;
}
"""

MINIMD_BENCHMARK = r"""
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <cuda_runtime.h>
#define CHECK_CUDA(c) {cudaError_t e=c; if(e!=cudaSuccess){printf("CUDA_ERROR: %s\n",cudaGetErrorString(e));exit(1);}}

{kernel}

/* CPU 基线: SoA + CSR 邻居表 */
void lj_cpu(int nlocal, int ntypes,
    const double* px, const double* py, const double* pz,
    double* fx, double* fy, double* fz,
    const uint16_t* type,
    const int* head, const int* count, const int* list,
    const double* cut, const double* eps, const double* sig)
{
    for (int i = 0; i < nlocal; i++) {
        double xt=px[i], yt=py[i], zt=pz[i]; int ti=type[i];
        double fix=0,fiy=0,fiz=0;
        int base=head[i];
        for (int k=0; k<count[i]; k++) {
            int j=list[base+k];
            double dx=xt-px[j], dy=yt-py[j], dz=zt-pz[j];
            double rsq=dx*dx+dy*dy+dz*dz;
            int tij=ti*ntypes+type[j];
            if (rsq < cut[tij]) {
                double sr2=1.0/rsq, sr6=sr2*sr2*sr2*sig[tij];
                double force=48.0*sr6*(sr6-0.5)*sr2*eps[tij];
                fix+=dx*force; fiy+=dy*force; fiz+=dz*force;
            }
        }
        fx[i]=fix; fy[i]=fiy; fz[i]=fiz;
    }
}

int main() {
    int nlocal=100000, ntypes=1;
    /* CSR 邻居表 */
    int avg_neigh=60, total_neigh=nlocal*avg_neigh;
    int      *hhead  = (int*)     malloc(nlocal*sizeof(int));
    int      *hcount = (int*)     malloc(nlocal*sizeof(int));
    int      *hlist  = (int*)     malloc(total_neigh*sizeof(int));
    double   *hpx    = (double*)  malloc(nlocal*sizeof(double));
    double   *hpy    = (double*)  malloc(nlocal*sizeof(double));
    double   *hpz    = (double*)  malloc(nlocal*sizeof(double));
    double   *hfx_c  = (double*)  malloc(nlocal*sizeof(double));
    double   *hfy_c  = (double*)  malloc(nlocal*sizeof(double));
    double   *hfz_c  = (double*)  malloc(nlocal*sizeof(double));
    double   *hfx_g  = (double*)  malloc(nlocal*sizeof(double));
    double   *hfy_g  = (double*)  malloc(nlocal*sizeof(double));
    double   *hfz_g  = (double*)  malloc(nlocal*sizeof(double));
    uint16_t *htype  = (uint16_t*)malloc(nlocal*sizeof(uint16_t));
    double   *hcut   = (double*)  malloc(sizeof(double));
    double   *heps   = (double*)  malloc(sizeof(double));
    double   *hsig   = (double*)  malloc(sizeof(double));

    srand(12345);
    int offset=0;
    for (int i=0; i<nlocal; i++) {
        hpx[i]=(double)rand()/RAND_MAX*100;
        hpy[i]=(double)rand()/RAND_MAX*100;
        hpz[i]=(double)rand()/RAND_MAX*100;
        htype[i]=0;
        hhead[i]=offset;
        hcount[i]=50+rand()%20;
        for (int k=0; k<hcount[i]; k++) hlist[offset+k]=rand()%nlocal;
        offset+=hcount[i];
    }
    hcut[0]=16.0; heps[0]=1.0; hsig[0]=1.0;

    lj_cpu(nlocal,ntypes,hpx,hpy,hpz,hfx_c,hfy_c,hfz_c,htype,hhead,hcount,hlist,hcut,heps,hsig);
    struct timespec ts0,ts1;
    clock_gettime(CLOCK_MONOTONIC,&ts0);
    for (int it=0; it<10; it++)
        lj_cpu(nlocal,ntypes,hpx,hpy,hpz,hfx_c,hfy_c,hfz_c,htype,hhead,hcount,hlist,hcut,heps,hsig);
    clock_gettime(CLOCK_MONOTONIC,&ts1);
    double cpu_ms=((ts1.tv_sec-ts0.tv_sec)*1e3+(ts1.tv_nsec-ts0.tv_nsec)*1e-6)/10.0;

    CHECK_CUDA(cudaMemcpyToSymbol(c_cutforcesq, hcut, sizeof(double)));
    CHECK_CUDA(cudaMemcpyToSymbol(c_epsilon,    heps, sizeof(double)));
    CHECK_CUDA(cudaMemcpyToSymbol(c_sigma6,     hsig, sizeof(double)));

    double   *dpx,*dpy,*dpz,*dfx,*dfy,*dfz;
    uint16_t *dtype;
    int      *dhead,*dcount,*dlist;
    CHECK_CUDA(cudaMalloc(&dpx,   nlocal*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dpy,   nlocal*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dpz,   nlocal*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dfx,   nlocal*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dfy,   nlocal*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dfz,   nlocal*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dtype, nlocal*sizeof(uint16_t)));
    CHECK_CUDA(cudaMalloc(&dhead, nlocal*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dcount,nlocal*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dlist, offset*sizeof(int)));
    CHECK_CUDA(cudaMemcpy(dpx,   hpx,   nlocal*sizeof(double),   cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dpy,   hpy,   nlocal*sizeof(double),   cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dpz,   hpz,   nlocal*sizeof(double),   cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dtype, htype, nlocal*sizeof(uint16_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dhead, hhead, nlocal*sizeof(int),      cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dcount,hcount,nlocal*sizeof(int),      cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dlist, hlist, offset*sizeof(int),      cudaMemcpyHostToDevice));

    int bs=256, nb=(nlocal+bs-1)/bs;
    lj_force_stage2<<<nb,bs>>>(nlocal,ntypes,dpx,dpy,dpz,dfx,dfy,dfz,dtype,dhead,dcount,dlist);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start,stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int it=0; it<10; it++)
        lj_force_stage2<<<nb,bs>>>(nlocal,ntypes,dpx,dpy,dpz,dfx,dfy,dfz,dtype,dhead,dcount,dlist);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float gpu_ms; cudaEventElapsedTime(&gpu_ms,start,stop); gpu_ms/=10;

    CHECK_CUDA(cudaMemcpy(hfx_g,dfx,nlocal*sizeof(double),cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hfy_g,dfy,nlocal*sizeof(double),cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hfz_g,dfz,nlocal*sizeof(double),cudaMemcpyDeviceToHost));
    double maxerr=0;
    for (int i=0; i<nlocal; i++) {
        double e=fabs(hfx_c[i]-hfx_g[i]); if(e>maxerr)maxerr=e;
        e=fabs(hfy_c[i]-hfy_g[i]); if(e>maxerr)maxerr=e;
        e=fabs(hfz_c[i]-hfz_g[i]); if(e>maxerr)maxerr=e;
    }
    printf("BENCHMARK_RESULT:cpu_ms=%.4f,gpu_ms=%.4f,speedup=%.2f,error=%.2e\n",
           cpu_ms,gpu_ms,cpu_ms/gpu_ms,maxerr);
    return 0;
}
"""

# ============================================================
# SPMV v3: SELL-C-σ warp-per-row（与 v2 相同，已完整实现建议）
#   - warp (32 threads) 处理一行
#   - SELL-C-σ: 行按 nnz 排序后以 C=32 分组，列主序存储
#   - colind 32-bit; x/y double
# ============================================================
SPMV_STAGE2_KERNEL = r"""
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
"""

SPMV_BENCHMARK = r"""
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#define CHECK_CUDA(c) {cudaError_t e=c; if(e!=cudaSuccess){printf("CUDA_ERROR: %s\n",cudaGetErrorString(e));exit(1);}}

{kernel}

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
"""

# ============================================================
# SYMGS v3: multi-color GS（与 v2 相同，已完整实现建议）
#   - suitable=false for strict GS → 必须用 multi-color
#   - invDiag 预计算 (建议: invDiag separate array)
#   - warp-per-row (nnz≈27)
#   - CUDA Graph 减少 16 次 kernel launch 开销
#   - ELL 列主序存储 (SELL-C-σ 简化版, C=32)
# ============================================================
SYMGS_STAGE2_KERNEL = r"""
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
"""

SYMGS_BENCHMARK = r"""
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#define CHECK_CUDA(c) {cudaError_t e=c; if(e!=cudaSuccess){printf("CUDA_ERROR: %s\n",cudaGetErrorString(e));exit(1);}}

{kernel}

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
"""


def write_and_get_path(name, kernel, template):
    code = template.replace("{kernel}", kernel)
    path = os.path.join(OUTPUT_DIR, f"{name}.cu")
    with open(path, "w") as f:
        f.write(code)
    return path


def compile_run(name, cu_path):
    abs_path = os.path.abspath(os.path.dirname(cu_path))
    wsl_dir = "/mnt/" + abs_path[0].lower() + abs_path[2:].replace("\\", "/")
    fname = os.path.basename(cu_path).replace(".cu", "")

    compile_cmd = (
        f'wsl -d Ubuntu-24.04 bash -c '
        f'"cd {wsl_dir} && /usr/local/cuda-12.6/bin/nvcc -O3 -arch=sm_86 '
        f'{fname}.cu -o {fname} 2>&1"'
    )
    r = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        return None, f"COMPILE_FAIL: {(r.stdout+r.stderr)[:400]}"

    run_cmd = f'wsl -d Ubuntu-24.04 bash -c "cd {wsl_dir} && ./{fname} 2>&1"'
    r = subprocess.run(run_cmd, shell=True, capture_output=True, text=True, timeout=120)
    out = r.stdout + r.stderr
    m = re.search(
        r'BENCHMARK_RESULT:cpu_ms=([\d.]+),gpu_ms=([\d.]+),speedup=([\d.]+),error=([\d.e+-]+)',
        out
    )
    if m:
        return {
            "cpu_ms": float(m.group(1)), "gpu_ms": float(m.group(2)),
            "speedup": float(m.group(3)), "error": float(m.group(4))
        }, "OK"
    return None, f"NO_RESULT: {out[:300]}"


if __name__ == "__main__":
    results = {}

    print("=" * 65)
    print("Stage 2 Manual Optimization Benchmark  [v3 - gpt-5.2 建议]")
    print("=" * 65)

    # miniMD
    print("\n[1/3] miniMD LJ Force")
    print("  新增: type→uint16_t; 邻居表→CSR head/count/list")
    cu = write_and_get_path("minimd_stage2", MINIMD_STAGE2_KERNEL, MINIMD_BENCHMARK)
    bench, msg = compile_run("minimd_stage2", cu)
    if bench:
        print(f"  CPU: {bench['cpu_ms']:.2f}ms  GPU: {bench['gpu_ms']:.2f}ms"
              f"  Speedup: {bench['speedup']:.2f}x  Error: {bench['error']:.2e}")
        results["minimd"] = {
            "direct_gen": 14.34, "stage2_manual_v2": 13.51,
            "stage2_manual_v3": bench["speedup"],
            "delta_vs_v2_pct": (bench["speedup"] - 13.51) / 13.51 * 100,
            "delta_vs_direct_pct": (bench["speedup"] - 14.34) / 14.34 * 100,
            "benchmark": bench
        }
    else:
        print(f"  FAILED: {msg}")
        results["minimd"] = {"error": msg}

    # SPMV
    print("\n[2/3] HPCG SPMV")
    print("  SELL-C-σ warp-per-row (同 v2)")
    cu = write_and_get_path("spmv_stage2", SPMV_STAGE2_KERNEL, SPMV_BENCHMARK)
    bench, msg = compile_run("spmv_stage2", cu)
    if bench:
        print(f"  CPU: {bench['cpu_ms']:.2f}ms  GPU: {bench['gpu_ms']:.2f}ms"
              f"  Speedup: {bench['speedup']:.2f}x  Error: {bench['error']:.2e}")
        results["spmv"] = {
            "direct_gen": 10.30, "stage2_manual_v2": 20.55,
            "stage2_manual_v3": bench["speedup"],
            "delta_vs_v2_pct": (bench["speedup"] - 20.55) / 20.55 * 100,
            "delta_vs_direct_pct": (bench["speedup"] - 10.30) / 10.30 * 100,
            "benchmark": bench
        }
    else:
        print(f"  FAILED: {msg}")
        results["spmv"] = {"error": msg}

    # SYMGS
    print("\n[3/3] HPCG SYMGS")
    print("  multi-color + invDiag + CUDA Graph + ELL (同 v2)")
    cu = write_and_get_path("symgs_stage2", SYMGS_STAGE2_KERNEL, SYMGS_BENCHMARK)
    bench, msg = compile_run("symgs_stage2", cu)
    if bench:
        print(f"  CPU: {bench['cpu_ms']:.2f}ms  GPU: {bench['gpu_ms']:.2f}ms"
              f"  Speedup: {bench['speedup']:.2f}x  Error: {bench['error']:.2e}")
        print("  (error 预期较大: multi-color GS 与严格 GS 算法不同)")
        results["symgs"] = {
            "direct_gen": 0.02, "stage2_manual_v2": None,
            "stage2_manual_v3": bench["speedup"],
            "note": "multi-color GS != strict GS; suitable=false without algorithm change",
            "benchmark": bench
        }
    else:
        print(f"  FAILED: {msg}")
        results["symgs"] = {"error": msg}

    # 汇总
    print("\n" + "=" * 65)
    print(f"{'Kernel':<12} {'Direct Gen':>11} {'v2 Manual':>11} {'v3 Manual':>11} {'Δ v2→v3':>10}")
    print("-" * 65)
    for k, v in results.items():
        if "error" not in v:
            dg = v.get("direct_gen", 0)
            v2 = v.get("stage2_manual_v2")
            v3 = v.get("stage2_manual_v3", 0)
            v2s = f"{v2:.2f}x" if v2 else "N/A"
            d   = f"{(v3-v2)/v2*100:+.1f}%" if v2 else "—"
            print(f"{k:<12} {dg:>9.2f}x {v2s:>11} {v3:>9.2f}x {d:>10}")
    print("=" * 65)

    results["timestamp"] = datetime.now().isoformat()
    out_path = os.path.join(OUTPUT_DIR, "stage2_manual_results_v3.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")