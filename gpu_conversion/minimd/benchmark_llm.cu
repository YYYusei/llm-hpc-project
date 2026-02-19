#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

#define PAD 4
#define CHECK_CUDA(call) { cudaError_t e=call; if(e!=cudaSuccess){printf("CUDA err: %s\n",cudaGetErrorString(e));exit(1);} }

void force_lj_cpu(int nlocal, int ntypes, const double* x, double* f, const int* type,
    const int* neighbors, const int* numneigh, int maxneighs,
    const double* cutforcesq, const double* epsilon, const double* sigma6) {
    for(int i = 0; i < nlocal; i++) f[i*PAD+0]=f[i*PAD+1]=f[i*PAD+2]=0.0;
    for(int i = 0; i < nlocal; i++) {
        double xtmp=x[i*PAD+0],ytmp=x[i*PAD+1],ztmp=x[i*PAD+2];
        int type_i=type[i]; double fix=0,fiy=0,fiz=0;
        for(int k = 0; k < numneigh[i]; k++) {
            int j=neighbors[i*maxneighs+k];
            double delx=xtmp-x[j*PAD+0],dely=ytmp-x[j*PAD+1],delz=ztmp-x[j*PAD+2];
            double rsq=delx*delx+dely*dely+delz*delz;
            int type_ij=type_i*ntypes+type[j];
            if(rsq<cutforcesq[type_ij]) {
                double sr2=1.0/rsq,sr6=sr2*sr2*sr2*sigma6[type_ij];
                double force=48.0*sr6*(sr6-0.5)*sr2*epsilon[type_ij];
                fix+=delx*force; fiy+=dely*force; fiz+=delz*force;
            }
        }
        f[i*PAD+0]=fix; f[i*PAD+1]=fiy; f[i*PAD+2]=fiz;
    }
}

// GPT-5.2 Generated Kernel
__global__ void compute_fullneigh_kernel(int nlocal, int ntypes,
    const double* __restrict__ x, double* __restrict__ f, const int* __restrict__ type,
    const int* __restrict__ neighbors, const int* __restrict__ numneigh, int maxneighs,
    const double* __restrict__ cutforcesq, const double* __restrict__ epsilon,
    const double* __restrict__ sigma6) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nlocal) return;
    double xtmp=x[i*PAD+0],ytmp=x[i*PAD+1],ztmp=x[i*PAD+2];
    int type_i=type[i]; double fix=0,fiy=0,fiz=0;
    int nneigh=numneigh[i], base=i*maxneighs;
    for(int k=0; k<nneigh; k++) {
        int j=neighbors[base+k];
        double delx=xtmp-x[j*PAD+0],dely=ytmp-x[j*PAD+1],delz=ztmp-x[j*PAD+2];
        double rsq=delx*delx+dely*dely+delz*delz;
        int type_ij=type_i*ntypes+type[j];
        if(rsq<cutforcesq[type_ij]) {
            double sr2=1.0/rsq,sr6=sr2*sr2*sr2*sigma6[type_ij];
            double force=48.0*sr6*(sr6-0.5)*sr2*epsilon[type_ij];
            fix+=delx*force; fiy+=dely*force; fiz+=delz*force;
        }
    }
    f[i*PAD+0]=fix; f[i*PAD+1]=fiy; f[i*PAD+2]=fiz;
}

int main() {
    int sizes[]={10000,50000,100000,200000}; int ns=4;
    int ntypes=2, maxneighs=100;
    
    printf("\n==========================================\n");
    printf("   GPT-5.2 Kernel Benchmark (RTX 3060)\n");
    printf("==========================================\n");
    printf("%-10s %-10s %-10s %-10s\n","Atoms","CPU(ms)","GPU(ms)","Speedup");
    printf("------------------------------------------\n");
    
    for(int s=0; s<ns; s++) {
        int nlocal=sizes[s], nall=nlocal+nlocal/10;
        double *h_x=(double*)malloc(nall*PAD*sizeof(double));
        double *h_f=(double*)malloc(nlocal*PAD*sizeof(double));
        int *h_type=(int*)malloc(nall*sizeof(int));
        int *h_neigh=(int*)malloc(nlocal*maxneighs*sizeof(int));
        int *h_numn=(int*)malloc(nlocal*sizeof(int));
        double *h_cut=(double*)malloc(ntypes*ntypes*sizeof(double));
        double *h_eps=(double*)malloc(ntypes*ntypes*sizeof(double));
        double *h_sig=(double*)malloc(ntypes*ntypes*sizeof(double));
        
        srand(12345);
        for(int i=0;i<nall;i++){h_x[i*PAD+0]=(double)rand()/RAND_MAX*100;h_x[i*PAD+1]=(double)rand()/RAND_MAX*100;h_x[i*PAD+2]=(double)rand()/RAND_MAX*100;h_type[i]=rand()%ntypes;}
        for(int i=0;i<nlocal;i++){h_numn[i]=30+rand()%20;for(int k=0;k<h_numn[i];k++)h_neigh[i*maxneighs+k]=rand()%nall;}
        for(int i=0;i<ntypes*ntypes;i++){h_cut[i]=6.25;h_eps[i]=1.0;h_sig[i]=1.0;}
        
        clock_t t1=clock();
        for(int r=0;r<5;r++) force_lj_cpu(nlocal,ntypes,h_x,h_f,h_type,h_neigh,h_numn,maxneighs,h_cut,h_eps,h_sig);
        double cpu_ms=(double)(clock()-t1)/CLOCKS_PER_SEC*1000/5;
        
        double *d_x,*d_f,*d_cut,*d_eps,*d_sig; int *d_type,*d_neigh,*d_numn;
        CHECK_CUDA(cudaMalloc(&d_x,nall*PAD*sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_f,nlocal*PAD*sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_type,nall*sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_neigh,nlocal*maxneighs*sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_numn,nlocal*sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_cut,ntypes*ntypes*sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_eps,ntypes*ntypes*sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_sig,ntypes*ntypes*sizeof(double)));
        CHECK_CUDA(cudaMemcpy(d_x,h_x,nall*PAD*sizeof(double),cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_type,h_type,nall*sizeof(int),cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_neigh,h_neigh,nlocal*maxneighs*sizeof(int),cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_numn,h_numn,nlocal*sizeof(int),cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_cut,h_cut,ntypes*ntypes*sizeof(double),cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_eps,h_eps,ntypes*ntypes*sizeof(double),cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_sig,h_sig,ntypes*ntypes*sizeof(double),cudaMemcpyHostToDevice));
        
        int bs=256, nb=(nlocal+bs-1)/bs;
        compute_fullneigh_kernel<<<nb,bs>>>(nlocal,ntypes,d_x,d_f,d_type,d_neigh,d_numn,maxneighs,d_cut,d_eps,d_sig);
        cudaDeviceSynchronize();
        
        cudaEvent_t start,stop; cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start);
        for(int r=0;r<10;r++) compute_fullneigh_kernel<<<nb,bs>>>(nlocal,ntypes,d_x,d_f,d_type,d_neigh,d_numn,maxneighs,d_cut,d_eps,d_sig);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float gpu_ms; cudaEventElapsedTime(&gpu_ms,start,stop); gpu_ms/=10;
        
        printf("%-10d %-10.2f %-10.2f %-10.1fx\n",nlocal,cpu_ms,gpu_ms,cpu_ms/gpu_ms);
        
        free(h_x);free(h_f);free(h_type);free(h_neigh);free(h_numn);free(h_cut);free(h_eps);free(h_sig);
        cudaFree(d_x);cudaFree(d_f);cudaFree(d_type);cudaFree(d_neigh);cudaFree(d_numn);cudaFree(d_cut);cudaFree(d_eps);cudaFree(d_sig);
    }
    printf("==========================================\n");
    return 0;
}
