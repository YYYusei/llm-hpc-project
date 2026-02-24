
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define PAD 4
#define CHECK_CUDA(call) { cudaError_t e=call; if(e!=cudaSuccess){printf("CUDA_ERROR: %s\n",cudaGetErrorString(e));exit(1);} }

// Helper functions for __ldg
__device__ __forceinline__ int ldg_int(const int* p) { return __ldg(p); }
__device__ __forceinline__ double ldg_double(const double* p) { return __ldg(p); }

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


// ============ Optimized Kernel ============
// Fixed kernel code here
__global__ void optimized_kernel_gpt_4o(int nlocal, int ntypes,
    const double* __restrict__ x, double* __restrict__ f, const int* __restrict__ type,
    const int* __restrict__ neighbors, const int* __restrict__ numneigh, int maxneighs,
    const double* __restrict__ cutforcesq, const double* __restrict__ epsilon,
    const double* __restrict__ sigma6)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nlocal) return;

    // Load position and type into registers
    const double xtmp = x[i * 4 + 0];
    const double ytmp = x[i * 4 + 1];
    const double ztmp = x[i * 4 + 2];
    const int type_i  = type[i];

    // Force accumulators
    double fix = 0.0, fiy = 0.0, fiz = 0.0;

    const int nneigh = numneigh[i];
    const int base   = i * maxneighs;

    // BUGFIX:
    // The original code used "extern __shared__ double shared_x[]" and then indexed it as
    // shared_x[tid*4 + 0..3] for tid in [0, blockDim.x). That requires 4*blockDim.x doubles
    // of dynamic shared memory. If the launch doesn't provide enough shared memory, this
    // causes out-of-bounds shared memory writes -> undefined behavior / hang / watchdog timeout.
    //
    // Fix: use statically-sized shared memory for a known maximum block size, and store
    // neighbor type as int (not double) to avoid type punning and wasted bandwidth.
    // This keeps the caching optimization but removes the dependency on dynamic shared memory.
    constexpr int MAX_BLOCK = 1024; // supports up to 1024 threads/block
    __shared__ double shx[MAX_BLOCK];
    __shared__ double shy[MAX_BLOCK];
    __shared__ double shz[MAX_BLOCK];
    __shared__ int    sht[MAX_BLOCK];

    const int tid = threadIdx.x;

    // Loop over neighbors in tiles of blockDim.x
    for (int k = 0; k < nneigh; k += blockDim.x) {

        // Load one neighbor per thread into shared memory (if in range)
        const int idx = k + tid;
        if (idx < nneigh) {
            const int j = neighbors[base + idx];
            shx[tid] = x[j * 4 + 0];
            shy[tid] = x[j * 4 + 1];
            shz[tid] = x[j * 4 + 2];
            sht[tid] = __ldg(&type[j]);
        }
        __syncthreads();

        // Process the loaded tile
        const int tileCount = min(blockDim.x, nneigh - k);
        #pragma unroll 4
        for (int l = 0; l < tileCount; ++l) {
            const double delx = xtmp - shx[l];
            const double dely = ytmp - shy[l];
            const double delz = ztmp - shz[l];
            const double rsq  = delx * delx + dely * dely + delz * delz;

            const int type_j  = sht[l];
            const int type_ij = type_i * ntypes + type_j;

            const double cutsq = __ldg(&cutforcesq[type_ij]);
            if (rsq < cutsq) {
                // (Assumes rsq > 0 for valid neighbor lists; if self can appear, add rsq>0 guard)
                const double sr2   = 1.0 / rsq;
                const double sr6   = (sr2 * sr2 * sr2) * __ldg(&sigma6[type_ij]);
                const double force = 48.0 * sr6 * (sr6 - 0.5) * sr2 * __ldg(&epsilon[type_ij]);

                fix = fma(delx, force, fix);
                fiy = fma(dely, force, fiy);
                fiz = fma(delz, force, fiz);
            }
        }
        __syncthreads();
    }

    // Store results
    f[i * 4 + 0] = fix;
    f[i * 4 + 1] = fiy;
    f[i * 4 + 2] = fiz;
}

// ============ CPU Reference ============
void cpu_force(int nlocal, int ntypes, const double* x, double* f, const int* type,
    const int* neighbors, const int* numneigh, int maxneighs,
    const double* cutforcesq, const double* epsilon, const double* sigma6) {
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
    int nlocal = 100000;
    int nall = nlocal + nlocal/10;
    int ntypes = 2, maxneighs = 100;
    
    // Allocate host memory
    double *h_x = (double*)malloc(nall * PAD * sizeof(double));
    double *h_f_cpu = (double*)malloc(nlocal * PAD * sizeof(double));
    double *h_f_bl = (double*)malloc(nlocal * PAD * sizeof(double));
    double *h_f_opt = (double*)malloc(nlocal * PAD * sizeof(double));
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
    
    // Benchmark baseline
    baseline_kernel<<<nb, bs>>>(nlocal, ntypes, d_x, d_f, d_type, d_neigh, d_numn, maxneighs, d_cut, d_eps, d_sig);
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaEventRecord(start);
    for(int r = 0; r < 10; r++)
        baseline_kernel<<<nb, bs>>>(nlocal, ntypes, d_x, d_f, d_type, d_neigh, d_numn, maxneighs, d_cut, d_eps, d_sig);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float bl_ms; cudaEventElapsedTime(&bl_ms, start, stop); bl_ms /= 10;
    CHECK_CUDA(cudaMemcpy(h_f_bl, d_f, nlocal * PAD * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Benchmark optimized
    optimized_kernel_gpt_4o<<<nb, bs>>>(nlocal, ntypes, d_x, d_f, d_type, d_neigh, d_numn, maxneighs, d_cut, d_eps, d_sig);
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaEventRecord(start);
    for(int r = 0; r < 10; r++)
        optimized_kernel_gpt_4o<<<nb, bs>>>(nlocal, ntypes, d_x, d_f, d_type, d_neigh, d_numn, maxneighs, d_cut, d_eps, d_sig);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float opt_ms; cudaEventElapsedTime(&opt_ms, start, stop); opt_ms /= 10;
    CHECK_CUDA(cudaMemcpy(h_f_opt, d_f, nlocal * PAD * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Check correctness
    double err = check_correctness(h_f_cpu, h_f_opt, nlocal);
    
    // Output results as JSON-like format for parsing
    printf("BENCHMARK_RESULT:baseline_ms=%.4f,optimized_ms=%.4f,speedup=%.2f,error=%.2e\n", 
           bl_ms, opt_ms, bl_ms/opt_ms, err);
    
    // Cleanup
    free(h_x); free(h_f_cpu); free(h_f_bl); free(h_f_opt);
    free(h_type); free(h_neigh); free(h_numn); free(h_cut); free(h_eps); free(h_sig);
    cudaFree(d_x); cudaFree(d_f); cudaFree(d_type); cudaFree(d_neigh);
    cudaFree(d_numn); cudaFree(d_cut); cudaFree(d_eps); cudaFree(d_sig);
    
    return 0;
}
