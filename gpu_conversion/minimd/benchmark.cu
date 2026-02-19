#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

#define PAD 4
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
}

void force_lj_cpu(int nlocal, int ntypes, const double* x, double* f, const int* type,
    const int* neighbors, const int* numneigh, int maxneighs,
    const double* cutforcesq, const double* epsilon, const double* sigma6)
{
    for(int i = 0; i < nlocal; i++) {
        f[i*PAD+0] = f[i*PAD+1] = f[i*PAD+2] = 0.0;
    }
    for(int i = 0; i < nlocal; i++) {
        double xtmp = x[i*PAD+0], ytmp = x[i*PAD+1], ztmp = x[i*PAD+2];
        int type_i = type[i];
        double fix = 0, fiy = 0, fiz = 0;
        for(int k = 0; k < numneigh[i]; k++) {
            int j = neighbors[i*maxneighs + k];
            double delx = xtmp - x[j*PAD+0];
            double dely = ytmp - x[j*PAD+1];
            double delz = ztmp - x[j*PAD+2];
            double rsq = delx*delx + dely*dely + delz*delz;
            int type_ij = type_i * ntypes + type[j];
            if(rsq < cutforcesq[type_ij] && rsq > 0.01) {
                double sr2 = 1.0 / rsq;
                double sr6 = sr2 * sr2 * sr2 * sigma6[type_ij];
                double force = 48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon[type_ij];
                fix += delx * force;
                fiy += dely * force;
                fiz += delz * force;
            }
        }
        f[i*PAD+0] = fix; f[i*PAD+1] = fiy; f[i*PAD+2] = fiz;
    }
}

__global__ void force_lj_kernel(int nlocal, int ntypes, const double* x, double* f, const int* type,
    const int* neighbors, const int* numneigh, int maxneighs,
    const double* cutforcesq, const double* epsilon, const double* sigma6)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= nlocal) return;
    
    double xtmp = x[i*PAD+0], ytmp = x[i*PAD+1], ztmp = x[i*PAD+2];
    int type_i = type[i];
    double fix = 0, fiy = 0, fiz = 0;
    
    for(int k = 0; k < numneigh[i]; k++) {
        int j = neighbors[i*maxneighs + k];
        double delx = xtmp - x[j*PAD+0];
        double dely = ytmp - x[j*PAD+1];
        double delz = ztmp - x[j*PAD+2];
        double rsq = delx*delx + dely*dely + delz*delz;
        int type_ij = type_i * ntypes + type[j];
        if(rsq < cutforcesq[type_ij] && rsq > 0.01) {
            double sr2 = 1.0 / rsq;
            double sr6 = sr2 * sr2 * sr2 * sigma6[type_ij];
            double force = 48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon[type_ij];
            fix += delx * force;
            fiy += dely * force;
            fiz += delz * force;
        }
    }
    f[i*PAD+0] = fix; f[i*PAD+1] = fiy; f[i*PAD+2] = fiz;
}

int main() {
    int sizes[] = {10000, 50000, 100000, 200000};
    int num_sizes = 4;
    int ntypes = 2, maxneighs = 100;
    
    printf("\n==========================================\n");
    printf("   miniMD GPU Benchmark (RTX 3060)\n");
    printf("==========================================\n");
    printf("%-10s %-12s %-12s %-10s\n", "Atoms", "CPU(ms)", "GPU(ms)", "Speedup");
    printf("------------------------------------------\n");
    
    for(int s = 0; s < num_sizes; s++) {
        int nlocal = sizes[s];
        int nall = nlocal + nlocal/10;
        
        double *h_x = (double*)malloc(nall * PAD * sizeof(double));
        double *h_f_cpu = (double*)malloc(nlocal * PAD * sizeof(double));
        double *h_f_gpu = (double*)malloc(nlocal * PAD * sizeof(double));
        int *h_type = (int*)malloc(nall * sizeof(int));
        int *h_neighbors = (int*)malloc(nlocal * maxneighs * sizeof(int));
        int *h_numneigh = (int*)malloc(nlocal * sizeof(int));
        double *h_cutforcesq = (double*)malloc(ntypes * ntypes * sizeof(double));
        double *h_epsilon = (double*)malloc(ntypes * ntypes * sizeof(double));
        double *h_sigma6 = (double*)malloc(ntypes * ntypes * sizeof(double));
        
        srand(12345);
        for(int i = 0; i < nall; i++) {
            h_x[i*PAD+0] = (double)rand()/RAND_MAX * 100.0;
            h_x[i*PAD+1] = (double)rand()/RAND_MAX * 100.0;
            h_x[i*PAD+2] = (double)rand()/RAND_MAX * 100.0;
            h_type[i] = rand() % ntypes;
        }
        for(int i = 0; i < nlocal; i++) {
            h_numneigh[i] = 30 + rand() % 20;
            for(int k = 0; k < h_numneigh[i]; k++)
                h_neighbors[i*maxneighs + k] = rand() % nall;
        }
        for(int i = 0; i < ntypes*ntypes; i++) {
            h_cutforcesq[i] = 6.25; h_epsilon[i] = 1.0; h_sigma6[i] = 1.0;
        }
        
        // CPU
        clock_t t1 = clock();
        for(int iter = 0; iter < 5; iter++)
            force_lj_cpu(nlocal, ntypes, h_x, h_f_cpu, h_type, h_neighbors, h_numneigh, maxneighs, h_cutforcesq, h_epsilon, h_sigma6);
        double cpu_time = (double)(clock() - t1) / CLOCKS_PER_SEC * 1000 / 5;
        
        // GPU
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
        CHECK_CUDA(cudaMemcpy(d_neigh, h_neighbors, nlocal * maxneighs * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_numn, h_numneigh, nlocal * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_cut, h_cutforcesq, ntypes * ntypes * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_eps, h_epsilon, ntypes * ntypes * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_sig, h_sigma6, ntypes * ntypes * sizeof(double), cudaMemcpyHostToDevice));
        
        int blockSize = 256;
        int numBlocks = (nlocal + blockSize - 1) / blockSize;
        
        // Warmup
        force_lj_kernel<<<numBlocks, blockSize>>>(nlocal, ntypes, d_x, d_f, d_type, d_neigh, d_numn, maxneighs, d_cut, d_eps, d_sig);
        cudaDeviceSynchronize();
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        for(int iter = 0; iter < 10; iter++)
            force_lj_kernel<<<numBlocks, blockSize>>>(nlocal, ntypes, d_x, d_f, d_type, d_neigh, d_numn, maxneighs, d_cut, d_eps, d_sig);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float gpu_ms;
        cudaEventElapsedTime(&gpu_ms, start, stop);
        double gpu_time = gpu_ms / 10;
        
        printf("%-10d %-12.3f %-12.3f %-10.2fx\n", nlocal, cpu_time, gpu_time, cpu_time/gpu_time);
        
        free(h_x); free(h_f_cpu); free(h_f_gpu); free(h_type);
        free(h_neighbors); free(h_numneigh);
        free(h_cutforcesq); free(h_epsilon); free(h_sigma6);
        cudaFree(d_x); cudaFree(d_f); cudaFree(d_type);
        cudaFree(d_neigh); cudaFree(d_numn);
        cudaFree(d_cut); cudaFree(d_eps); cudaFree(d_sig);
    }
    
    printf("==========================================\n");
    return 0;
}
