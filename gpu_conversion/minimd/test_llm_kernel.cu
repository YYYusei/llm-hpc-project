#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define PAD 4
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// CPU reference
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
            if(rsq < cutforcesq[type_ij]) {
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

// LLM Generated kernel (GPT-5.2)
__global__ void compute_fullneigh_kernel(
    int nlocal, int ntypes,
    const double* __restrict__ x,
    double* __restrict__ f,
    const int* __restrict__ type,
    const int* __restrict__ neighbors,
    const int* __restrict__ numneigh,
    int maxneighs,
    const double* __restrict__ cutforcesq,
    const double* __restrict__ epsilon,
    const double* __restrict__ sigma6)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nlocal) return;
    double xtmp = x[i * PAD + 0];
    double ytmp = x[i * PAD + 1];
    double ztmp = x[i * PAD + 2];
    int type_i = type[i];
    double fix = 0.0, fiy = 0.0, fiz = 0.0;
    int nneigh = numneigh[i];
    int base = i * maxneighs;
    for (int k = 0; k < nneigh; k++) {
        int j = neighbors[base + k];
        double delx = xtmp - x[j * PAD + 0];
        double dely = ytmp - x[j * PAD + 1];
        double delz = ztmp - x[j * PAD + 2];
        double rsq = delx * delx + dely * dely + delz * delz;
        int type_ij = type_i * ntypes + type[j];
        if (rsq < cutforcesq[type_ij]) {
            double sr2 = 1.0 / rsq;
            double sr6 = sr2 * sr2 * sr2 * sigma6[type_ij];
            double force = 48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon[type_ij];
            fix += delx * force;
            fiy += dely * force;
            fiz += delz * force;
        }
    }
    f[i * PAD + 0] = fix;
    f[i * PAD + 1] = fiy;
    f[i * PAD + 2] = fiz;
}

int main() {
    int nlocal = 50000, nall = 55000, ntypes = 2, maxneighs = 100;
    
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
    force_lj_cpu(nlocal, ntypes, h_x, h_f_cpu, h_type, h_neighbors, h_numneigh, maxneighs, h_cutforcesq, h_epsilon, h_sigma6);
    
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
    CHECK_CUDA(cudaMemset(d_f, 0, nlocal * PAD * sizeof(double)));
    
    int blockSize = 256;
    int numBlocks = (nlocal + blockSize - 1) / blockSize;
    
    compute_fullneigh_kernel<<<numBlocks, blockSize>>>(
        nlocal, ntypes, d_x, d_f, d_type, d_neigh, d_numn, maxneighs, d_cut, d_eps, d_sig);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaMemcpy(h_f_gpu, d_f, nlocal * PAD * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Verify
    double max_err = 0.0;
    for(int i = 0; i < nlocal; i++) {
        for(int d = 0; d < 3; d++) {
            double err = fabs(h_f_cpu[i*PAD+d] - h_f_gpu[i*PAD+d]);
            double rel = err / (fabs(h_f_cpu[i*PAD+d]) + 1e-10);
            if(rel > max_err) max_err = rel;
        }
    }
    
    printf("\n==========================================\n");
    printf("   GPT-5.2 Generated Kernel Test\n");
    printf("==========================================\n");
    printf("Atoms: %d\n", nlocal);
    printf("Max relative error: %.2e\n", max_err);
    printf("Verification: %s\n", max_err < 1e-6 ? "PASSED" : "FAILED");
    printf("==========================================\n");
    
    return max_err < 1e-6 ? 0 : 1;
}
