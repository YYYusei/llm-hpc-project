
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <cuda_runtime.h>
#define CHECK_CUDA(c) {cudaError_t e=c; if(e!=cudaSuccess){printf("CUDA_ERROR: %s\n",cudaGetErrorString(e));exit(1);}}


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
