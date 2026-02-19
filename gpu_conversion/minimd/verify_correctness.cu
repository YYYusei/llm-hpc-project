#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

#define PAD 4
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// CPU reference implementation
void force_lj_cpu(
    int nlocal, int ntypes,
    const double* x, double* f, const int* type,
    const int* neighbors, const int* numneigh, int maxneighs,
    const double* cutforcesq, const double* epsilon, const double* sigma6,
    double* eng_vdwl, double* virial)
{
    double t_eng = 0.0, t_vir = 0.0;
    
    for(int i = 0; i < nlocal; i++) {
        f[i*PAD+0] = 0.0;
        f[i*PAD+1] = 0.0;
        f[i*PAD+2] = 0.0;
    }
    
    for(int i = 0; i < nlocal; i++) {
        const double xtmp = x[i*PAD+0];
        const double ytmp = x[i*PAD+1];
        const double ztmp = x[i*PAD+2];
        const int type_i = type[i];
        
        double fix = 0.0, fiy = 0.0, fiz = 0.0;
        
        for(int k = 0; k < numneigh[i]; k++) {
            const int j = neighbors[i*maxneighs + k];
            const double delx = xtmp - x[j*PAD+0];
            const double dely = ytmp - x[j*PAD+1];
            const double delz = ztmp - x[j*PAD+2];
            const double rsq = delx*delx + dely*dely + delz*delz;
            
            const int type_ij = type_i * ntypes + type[j];
            
            if(rsq < cutforcesq[type_ij]) {
                const double sr2 = 1.0 / rsq;
                const double sr6 = sr2 * sr2 * sr2 * sigma6[type_ij];
                const double force = 48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon[type_ij];
                
                fix += delx * force;
                fiy += dely * force;
                fiz += delz * force;
                
                t_eng += sr6 * (sr6 - 1.0) * epsilon[type_ij];
                t_vir += rsq * force;
            }
        }
        
        f[i*PAD+0] = fix;
        f[i*PAD+1] = fiy;
        f[i*PAD+2] = fiz;
    }
    
    *eng_vdwl = t_eng * 4.0;
    *virial = t_vir * 0.5;
}

// GPU kernel - basic version
__global__ void force_lj_kernel(
    int nlocal, int ntypes,
    const double* x, double* f, const int* type,
    const int* neighbors, const int* numneigh, int maxneighs,
    const double* cutforcesq, const double* epsilon, const double* sigma6,
    double* eng_vdwl, double* virial)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= nlocal) return;
    
    const double xtmp = x[i*PAD+0];
    const double ytmp = x[i*PAD+1];
    const double ztmp = x[i*PAD+2];
    const int type_i = type[i];
    
    double fix = 0.0, fiy = 0.0, fiz = 0.0;
    double t_eng = 0.0, t_vir = 0.0;
    
    for(int k = 0; k < numneigh[i]; k++) {
        const int j = neighbors[i*maxneighs + k];
        const double delx = xtmp - x[j*PAD+0];
        const double dely = ytmp - x[j*PAD+1];
        const double delz = ztmp - x[j*PAD+2];
        const double rsq = delx*delx + dely*dely + delz*delz;
        
        const int type_ij = type_i * ntypes + type[j];
        
        if(rsq < cutforcesq[type_ij]) {
            const double sr2 = 1.0 / rsq;
            const double sr6 = sr2 * sr2 * sr2 * sigma6[type_ij];
            const double force = 48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon[type_ij];
            
            fix += delx * force;
            fiy += dely * force;
            fiz += delz * force;
            
            t_eng += sr6 * (sr6 - 1.0) * epsilon[type_ij];
            t_vir += rsq * force;
        }
    }
    
    f[i*PAD+0] = fix;
    f[i*PAD+1] = fiy;
    f[i*PAD+2] = fiz;
    
    atomicAdd(eng_vdwl, t_eng * 4.0);
    atomicAdd(virial, t_vir * 0.5);
}

void init_test_data(int nlocal, int nall, int ntypes, int maxneighs,
    double* x, int* type, int* neighbors, int* numneigh,
    double* cutforcesq, double* epsilon, double* sigma6)
{
    srand(12345);
    
    for(int i = 0; i < nall; i++) {
        x[i*PAD+0] = (double)rand()/RAND_MAX * 50.0;
        x[i*PAD+1] = (double)rand()/RAND_MAX * 50.0;
        x[i*PAD+2] = (double)rand()/RAND_MAX * 50.0;
        type[i] = rand() % ntypes;
    }
    
    for(int i = 0; i < nlocal; i++) {
        numneigh[i] = 20 + rand() % 30;
        for(int k = 0; k < numneigh[i]; k++) {
            neighbors[i*maxneighs + k] = rand() % nall;
        }
    }
    
    double cutforce = 2.5;
    for(int i = 0; i < ntypes*ntypes; i++) {
        cutforcesq[i] = cutforce * cutforce;
        epsilon[i] = 1.0;
        sigma6[i] = 1.0;
    }
}

int main() {
    int nlocal = 10000;
    int nall = nlocal + 1000;
    int ntypes = 2;
    int maxneighs = 100;
    
    // Host memory
    double *h_x = (double*)malloc(nall * PAD * sizeof(double));
    double *h_f_cpu = (double*)malloc(nlocal * PAD * sizeof(double));
    double *h_f_gpu = (double*)malloc(nlocal * PAD * sizeof(double));
    int *h_type = (int*)malloc(nall * sizeof(int));
    int *h_neighbors = (int*)malloc(nlocal * maxneighs * sizeof(int));
    int *h_numneigh = (int*)malloc(nlocal * sizeof(int));
    double *h_cutforcesq = (double*)malloc(ntypes * ntypes * sizeof(double));
    double *h_epsilon = (double*)malloc(ntypes * ntypes * sizeof(double));
    double *h_sigma6 = (double*)malloc(ntypes * ntypes * sizeof(double));
    
    init_test_data(nlocal, nall, ntypes, maxneighs,
        h_x, h_type, h_neighbors, h_numneigh,
        h_cutforcesq, h_epsilon, h_sigma6);
    
    // CPU computation
    double eng_cpu = 0, vir_cpu = 0;
    clock_t cpu_start = clock();
    force_lj_cpu(nlocal, ntypes, h_x, h_f_cpu, h_type,
        h_neighbors, h_numneigh, maxneighs,
        h_cutforcesq, h_epsilon, h_sigma6, &eng_cpu, &vir_cpu);
    clock_t cpu_end = clock();
    double cpu_time = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000;
    
    // Device memory
    double *d_x, *d_f, *d_cutforcesq, *d_epsilon, *d_sigma6, *d_eng, *d_vir;
    int *d_type, *d_neighbors, *d_numneigh;
    
    CHECK_CUDA(cudaMalloc(&d_x, nall * PAD * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_f, nlocal * PAD * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_type, nall * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_neighbors, nlocal * maxneighs * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_numneigh, nlocal * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_cutforcesq, ntypes * ntypes * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_epsilon, ntypes * ntypes * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_sigma6, ntypes * ntypes * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_eng, sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_vir, sizeof(double)));
    
    CHECK_CUDA(cudaMemcpy(d_x, h_x, nall * PAD * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_type, h_type, nall * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_neighbors, h_neighbors, nlocal * maxneighs * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_numneigh, h_numneigh, nlocal * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cutforcesq, h_cutforcesq, ntypes * ntypes * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_epsilon, h_epsilon, ntypes * ntypes * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_sigma6, h_sigma6, ntypes * ntypes * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_f, 0, nlocal * PAD * sizeof(double)));
    CHECK_CUDA(cudaMemset(d_eng, 0, sizeof(double)));
    CHECK_CUDA(cudaMemset(d_vir, 0, sizeof(double)));
    
    // GPU computation
    int blockSize = 256;
    int numBlocks = (nlocal + blockSize - 1) / blockSize;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    force_lj_kernel<<<numBlocks, blockSize>>>(
        nlocal, ntypes, d_x, d_f, d_type,
        d_neighbors, d_numneigh, maxneighs,
        d_cutforcesq, d_epsilon, d_sigma6, d_eng, d_vir);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);
    
    CHECK_CUDA(cudaMemcpy(h_f_gpu, d_f, nlocal * PAD * sizeof(double), cudaMemcpyDeviceToHost));
    double eng_gpu, vir_gpu;
    CHECK_CUDA(cudaMemcpy(&eng_gpu, d_eng, sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&vir_gpu, d_vir, sizeof(double), cudaMemcpyDeviceToHost));
    
    // Verify
    printf("\n========================================\n");
    printf("   miniMD Force LJ - GPU Verification\n");
    printf("========================================\n");
    printf("Atoms: %d, Avg neighbors: ~35\n\n", nlocal);
    
    double max_err = 0.0;
    for(int i = 0; i < nlocal; i++) {
        for(int d = 0; d < 3; d++) {
            double err = fabs(h_f_cpu[i*PAD+d] - h_f_gpu[i*PAD+d]);
            double rel = err / (fabs(h_f_cpu[i*PAD+d]) + 1e-10);
            if(rel > max_err) max_err = rel;
        }
    }
    
    printf("Force max relative error: %.2e\n", max_err);
    printf("Force verification: %s\n\n", max_err < 1e-6 ? "PASSED" : "FAILED");
    
    double eng_err = fabs(eng_cpu - eng_gpu) / (fabs(eng_cpu) + 1e-10);
    double vir_err = fabs(vir_cpu - vir_gpu) / (fabs(vir_cpu) + 1e-10);
    printf("Energy: CPU=%.6f, GPU=%.6f, err=%.2e\n", eng_cpu, eng_gpu, eng_err);
    printf("Virial: CPU=%.6f, GPU=%.6f, err=%.2e\n\n", vir_cpu, vir_gpu, vir_err);
    
    printf("CPU time: %.3f ms\n", cpu_time);
    printf("GPU time: %.3f ms\n", gpu_time);
    printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    printf("========================================\n");
    
    // Cleanup
    free(h_x); free(h_f_cpu); free(h_f_gpu);
    free(h_type); free(h_neighbors); free(h_numneigh);
    free(h_cutforcesq); free(h_epsilon); free(h_sigma6);
    cudaFree(d_x); cudaFree(d_f); cudaFree(d_type);
    cudaFree(d_neighbors); cudaFree(d_numneigh);
    cudaFree(d_cutforcesq); cudaFree(d_epsilon); cudaFree(d_sigma6);
    cudaFree(d_eng); cudaFree(d_vir);
    
    return max_err < 1e-6 ? 0 : 1;
}
