#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define PAD 4
#define CHECK_CUDA(call) { cudaError_t e=call; if(e!=cudaSuccess){printf("CUDA error: %s\n",cudaGetErrorString(e));exit(1);} }

// ============ Baseline Kernel ============
__global__ void baseline_kernel(int nlocal, int ntypes,
    const double* __restrict__ x, double* __restrict__ f, const int* __restrict__ type,
    const int* __restrict__ neighbors, const int* __restrict__ numneigh, int maxneighs,
    const double* __restrict__ cutforcesq, const double* __restrict__ epsilon,
    const double* __restrict__ sigma6) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nlocal) return;
    double xtmp=x[i*4+0], ytmp=x[i*4+1], ztmp=x[i*4+2];
    int type_i=type[i]; 
    double fix=0, fiy=0, fiz=0;
    int nneigh=numneigh[i], base=i*maxneighs;
    for(int k=0; k<nneigh; k++) {
        int j=neighbors[base+k];
        double delx=xtmp-x[j*4+0], dely=ytmp-x[j*4+1], delz=ztmp-x[j*4+2];
        double rsq=delx*delx+dely*dely+delz*delz;
        int type_ij=type_i*ntypes+type[j];
        if(rsq<cutforcesq[type_ij]) {
            double sr2=1.0/rsq, sr6=sr2*sr2*sr2*sigma6[type_ij];
            double force=48.0*sr6*(sr6-0.5)*sr2*epsilon[type_ij];
            fix+=delx*force; fiy+=dely*force; fiz+=delz*force;
        }
    }
    f[i*4+0]=fix; f[i*4+1]=fiy; f[i*4+2]=fiz;
}

// ============ GPT-4o Optimized Kernel ============
__global__ void gpt4o_kernel(int nlocal, int ntypes,
    const double* __restrict__ x, double* __restrict__ f, const int* __restrict__ type,
    const int* __restrict__ neighbors, const int* __restrict__ numneigh, int maxneighs,
    const double* __restrict__ cutforcesq, const double* __restrict__ epsilon,
    const double* __restrict__ sigma6) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nlocal) return;
    extern __shared__ double shared_data[];
    double* shared_x = shared_data;
    int* shared_type = (int*)&shared_x[blockDim.x * 4];
    double xtmp = x[i * 4 + 0];
    double ytmp = x[i * 4 + 1];
    double ztmp = x[i * 4 + 2];
    int type_i = type[i];
    double fix = 0.0, fiy = 0.0, fiz = 0.0;
    int nneigh = numneigh[i];
    int base = i * maxneighs;
    for (int k = threadIdx.x; k < nneigh; k += blockDim.x) {
        int j = neighbors[base + k];
        shared_x[k * 4 + 0] = x[j * 4 + 0];
        shared_x[k * 4 + 1] = x[j * 4 + 1];
        shared_x[k * 4 + 2] = x[j * 4 + 2];
        shared_type[k] = type[j];
    }
    __syncthreads();
    for (int k = 0; k < nneigh; k++) {
        double delx = xtmp - shared_x[k * 4 + 0];
        double dely = ytmp - shared_x[k * 4 + 1];
        double delz = ztmp - shared_x[k * 4 + 2];
        int type_j = shared_type[k];
        double rsq = delx * delx + dely * dely + delz * delz;
        int type_ij = type_i * ntypes + type_j;
        if (rsq < cutforcesq[type_ij]) {
            double sr2 = 1.0 / rsq;
            double sr6 = sr2 * sr2 * sr2 * sigma6[type_ij];
            double force = 48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon[type_ij];
            fix += delx * force;
            fiy += dely * force;
            fiz += delz * force;
        }
    }
    f[i * 4 + 0] = fix;
    f[i * 4 + 1] = fiy;
    f[i * 4 + 2] = fiz;
}

// ============ GPT-5.2 Optimized Kernel ============
__device__ __forceinline__ int ldg_int(const int* p) { return __ldg(p); }
__device__ __forceinline__ double ldg_double(const double* p) { return __ldg(p); }

__global__ void gpt52_kernel(int nlocal, int ntypes,
    const double* __restrict__ x, double* __restrict__ f, const int* __restrict__ type,
    const int* __restrict__ neighbors, const int* __restrict__ numneigh, int maxneighs,
    const double* __restrict__ cutforcesq, const double* __restrict__ epsilon,
    const double* __restrict__ sigma6) {
    
    __shared__ double4 sh_xi[256];
    __shared__ int sh_ti[256];
    
    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + tid;
    
    if (i < nlocal) {
        const double4 xi = reinterpret_cast<const double4*>(x)[i];
        sh_xi[tid] = xi;
        sh_ti[tid] = ldg_int(&type[i]);
    }
    __syncthreads();
    
    if (i >= nlocal) return;
    
    const double4 xi = sh_xi[tid];
    const int type_i = sh_ti[tid];
    double fix = 0.0, fiy = 0.0, fiz = 0.0;
    const int nneigh = ldg_int(&numneigh[i]);
    const int base = i * maxneighs;
    
    int k = 0;
    for (; k + 3 < nneigh; k += 4) {
        const int j0 = ldg_int(&neighbors[base + k + 0]);
        const int j1 = ldg_int(&neighbors[base + k + 1]);
        const int j2 = ldg_int(&neighbors[base + k + 2]);
        const int j3 = ldg_int(&neighbors[base + k + 3]);
        
        const double4 xj0 = reinterpret_cast<const double4*>(x)[j0];
        const double4 xj1 = reinterpret_cast<const double4*>(x)[j1];
        const double4 xj2 = reinterpret_cast<const double4*>(x)[j2];
        const double4 xj3 = reinterpret_cast<const double4*>(x)[j3];
        
        const int tj0 = ldg_int(&type[j0]);
        const int tj1 = ldg_int(&type[j1]);
        const int tj2 = ldg_int(&type[j2]);
        const int tj3 = ldg_int(&type[j3]);
        
        #define COMPUTE_FORCE(jx, tj) { \
            const double delx = xi.x - jx.x; \
            const double dely = xi.y - jx.y; \
            const double delz = xi.z - jx.z; \
            const double rsq = delx*delx + dely*dely + delz*delz; \
            const int type_ij = type_i * ntypes + tj; \
            const double cutsq = ldg_double(&cutforcesq[type_ij]); \
            if (rsq < cutsq) { \
                const double sr2 = 1.0/rsq; \
                const double s6 = ldg_double(&sigma6[type_ij]); \
                const double eps = ldg_double(&epsilon[type_ij]); \
                const double sr6 = (sr2*sr2*sr2) * s6; \
                const double force = (48.0*eps) * sr6 * (sr6-0.5) * sr2; \
                fix = fma(delx, force, fix); \
                fiy = fma(dely, force, fiy); \
                fiz = fma(delz, force, fiz); \
            } \
        }
        
        COMPUTE_FORCE(xj0, tj0);
        COMPUTE_FORCE(xj1, tj1);
        COMPUTE_FORCE(xj2, tj2);
        COMPUTE_FORCE(xj3, tj3);
        #undef COMPUTE_FORCE
    }
    
    for (; k < nneigh; k++) {
        const int j = ldg_int(&neighbors[base + k]);
        const double4 xj = reinterpret_cast<const double4*>(x)[j];
        const double delx = xi.x - xj.x;
        const double dely = xi.y - xj.y;
        const double delz = xi.z - xj.z;
        const double rsq = delx*delx + dely*dely + delz*delz;
        const int tj = ldg_int(&type[j]);
        const int type_ij = type_i * ntypes + tj;
        const double cutsq = ldg_double(&cutforcesq[type_ij]);
        if (rsq < cutsq) {
            const double sr2 = 1.0/rsq;
            const double s6 = ldg_double(&sigma6[type_ij]);
            const double eps = ldg_double(&epsilon[type_ij]);
            const double sr6 = (sr2*sr2*sr2) * s6;
            const double force = (48.0*eps) * sr6 * (sr6-0.5) * sr2;
            fix = fma(delx, force, fix);
            fiy = fma(dely, force, fiy);
            fiz = fma(delz, force, fiz);
        }
    }
    
    reinterpret_cast<double4*>(f)[i] = make_double4(fix, fiy, fiz, 0.0);
}

// ============ CPU Reference ============
void cpu_force(int nlocal, int ntypes, const double* x, double* f, const int* type,
    const int* neighbors, const int* numneigh, int maxneighs,
    const double* cutforcesq, const double* epsilon, const double* sigma6) {
    for(int i = 0; i < nlocal; i++) {
        f[i*PAD+0]=f[i*PAD+1]=f[i*PAD+2]=0.0;
    }
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

double check_correctness(double* f1, double* f2, int n) {
    double maxerr = 0.0;
    for(int i = 0; i < n; i++) {
        for(int c = 0; c < 3; c++) {
            double err = fabs(f1[i*PAD+c] - f2[i*PAD+c]);
            if(err > maxerr) maxerr = err;
        }
    }
    return maxerr;
}

int main() {
    int sizes[] = {50000, 100000, 200000};
    int ns = 3;
    int ntypes = 2, maxneighs = 100;
    
    printf("\n");
    printf("=================================================================\n");
    printf("   LLM-Generated CUDA Kernel Optimization Benchmark (RTX 3060)\n");
    printf("=================================================================\n");
    printf("%-10s %-12s %-12s %-12s %-10s %-10s\n", 
           "Atoms", "Baseline", "GPT-4o", "GPT-5.2", "4o vs BL", "5.2 vs BL");
    printf("-----------------------------------------------------------------\n");
    
    for(int s = 0; s < ns; s++) {
        int nlocal = sizes[s];
        int nall = nlocal + nlocal/10;
        
        // Allocate host memory
        double *h_x = (double*)malloc(nall * PAD * sizeof(double));
        double *h_f_cpu = (double*)malloc(nlocal * PAD * sizeof(double));
        double *h_f_bl = (double*)malloc(nlocal * PAD * sizeof(double));
        double *h_f_4o = (double*)malloc(nlocal * PAD * sizeof(double));
        double *h_f_52 = (double*)malloc(nlocal * PAD * sizeof(double));
        int *h_type = (int*)malloc(nall * sizeof(int));
        int *h_neigh = (int*)malloc(nlocal * maxneighs * sizeof(int));
        int *h_numn = (int*)malloc(nlocal * sizeof(int));
        double *h_cut = (double*)malloc(ntypes * ntypes * sizeof(double));
        double *h_eps = (double*)malloc(ntypes * ntypes * sizeof(double));
        double *h_sig = (double*)malloc(ntypes * ntypes * sizeof(double));
        
        // Initialize data
        srand(12345);
        for(int i = 0; i < nall; i++) {
            h_x[i*PAD+0] = (double)rand()/RAND_MAX * 100;
            h_x[i*PAD+1] = (double)rand()/RAND_MAX * 100;
            h_x[i*PAD+2] = (double)rand()/RAND_MAX * 100;
            h_x[i*PAD+3] = 0;
            h_type[i] = rand() % ntypes;
        }
        for(int i = 0; i < nlocal; i++) {
            h_numn[i] = 30 + rand() % 20;
            for(int k = 0; k < h_numn[i]; k++)
                h_neigh[i*maxneighs+k] = rand() % nall;
        }
        for(int i = 0; i < ntypes*ntypes; i++) {
            h_cut[i] = 6.25; h_eps[i] = 1.0; h_sig[i] = 1.0;
        }
        
        // CPU reference
        cpu_force(nlocal, ntypes, h_x, h_f_cpu, h_type, h_neigh, h_numn, maxneighs, h_cut, h_eps, h_sig);
        
        // Allocate device memory
        double *d_x, *d_f, *d_cut, *d_eps, *d_sig;
        int *d_type, *d_neigh, *d_numn;
        CHECK_CUDA(cudaMalloc(&d_x, nall * PAD * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_f, nlocal * PAD * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_type, nall * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_neigh, nlocal * maxneighs * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_numn, nlocal * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_cut, ntypes * ntypes * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_eps, ntypes * ntypes * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_sig, ntypes * ntypes * sizeof(double)));
        
        CHECK_CUDA(cudaMemcpy(d_x, h_x, nall * PAD * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_type, h_type, nall * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_neigh, h_neigh, nlocal * maxneighs * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_numn, h_numn, nlocal * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_cut, h_cut, ntypes * ntypes * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_eps, h_eps, ntypes * ntypes * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_sig, h_sig, ntypes * ntypes * sizeof(double), cudaMemcpyHostToDevice));
        
        int bs = 256, nb = (nlocal + bs - 1) / bs;
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        
        // Warmup and benchmark baseline
        baseline_kernel<<<nb, bs>>>(nlocal, ntypes, d_x, d_f, d_type, d_neigh, d_numn, maxneighs, d_cut, d_eps, d_sig);
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        for(int r = 0; r < 10; r++)
            baseline_kernel<<<nb, bs>>>(nlocal, ntypes, d_x, d_f, d_type, d_neigh, d_numn, maxneighs, d_cut, d_eps, d_sig);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float bl_ms; cudaEventElapsedTime(&bl_ms, start, stop); bl_ms /= 10;
        CHECK_CUDA(cudaMemcpy(h_f_bl, d_f, nlocal * PAD * sizeof(double), cudaMemcpyDeviceToHost));
        
        // Benchmark GPT-4o (with shared memory)
        size_t smem_4o = maxneighs * 4 * sizeof(double) + maxneighs * sizeof(int);
        gpt4o_kernel<<<nb, bs, smem_4o>>>(nlocal, ntypes, d_x, d_f, d_type, d_neigh, d_numn, maxneighs, d_cut, d_eps, d_sig);
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        for(int r = 0; r < 10; r++)
            gpt4o_kernel<<<nb, bs, smem_4o>>>(nlocal, ntypes, d_x, d_f, d_type, d_neigh, d_numn, maxneighs, d_cut, d_eps, d_sig);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms_4o; cudaEventElapsedTime(&ms_4o, start, stop); ms_4o /= 10;
        CHECK_CUDA(cudaMemcpy(h_f_4o, d_f, nlocal * PAD * sizeof(double), cudaMemcpyDeviceToHost));
        
        // Benchmark GPT-5.2
        gpt52_kernel<<<nb, bs>>>(nlocal, ntypes, d_x, d_f, d_type, d_neigh, d_numn, maxneighs, d_cut, d_eps, d_sig);
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        for(int r = 0; r < 10; r++)
            gpt52_kernel<<<nb, bs>>>(nlocal, ntypes, d_x, d_f, d_type, d_neigh, d_numn, maxneighs, d_cut, d_eps, d_sig);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms_52; cudaEventElapsedTime(&ms_52, start, stop); ms_52 /= 10;
        CHECK_CUDA(cudaMemcpy(h_f_52, d_f, nlocal * PAD * sizeof(double), cudaMemcpyDeviceToHost));
        
        // Check correctness
        double err_bl = check_correctness(h_f_cpu, h_f_bl, nlocal);
        double err_4o = check_correctness(h_f_cpu, h_f_4o, nlocal);
        double err_52 = check_correctness(h_f_cpu, h_f_52, nlocal);
        
        printf("%-10d %-12.3f %-12.3f %-12.3f %-10.2fx %-10.2fx\n",
               nlocal, bl_ms, ms_4o, ms_52, bl_ms/ms_4o, bl_ms/ms_52);
        
        if(err_4o > 1e-10 || err_52 > 1e-10) {
            printf("  [WARN] Error: baseline=%.2e, 4o=%.2e, 5.2=%.2e\n", err_bl, err_4o, err_52);
        }
        
        // Cleanup
        free(h_x); free(h_f_cpu); free(h_f_bl); free(h_f_4o); free(h_f_52);
        free(h_type); free(h_neigh); free(h_numn); free(h_cut); free(h_eps); free(h_sig);
        cudaFree(d_x); cudaFree(d_f); cudaFree(d_type); cudaFree(d_neigh);
        cudaFree(d_numn); cudaFree(d_cut); cudaFree(d_eps); cudaFree(d_sig);
    }
    
    printf("=================================================================\n");
    printf("Note: Times in ms. Speedup shows optimization vs baseline.\n");
    
    return 0;
}