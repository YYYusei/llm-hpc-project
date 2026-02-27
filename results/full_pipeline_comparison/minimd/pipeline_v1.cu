
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { cudaError_t e=call; if(e!=cudaSuccess){printf("CUDA_ERROR: %s\n",cudaGetErrorString(e));exit(1);} }

extern "C" __global__ void lj_force_pipeline(int nlocal, int ntypes,
    const double* __restrict__ x, double* __restrict__ f, const int* __restrict__ type,
    const int* __restrict__ neighbors, const int* __restrict__ numneigh, int maxneighs,
    const double* __restrict__ cutforcesq, const double* __restrict__ epsilon,
    const double* __restrict__ sigma6)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nlocal) return;

  // Vectorized load for x[i] (assumes PAD==4 layout: x[4*i + {0,1,2,3}])
  const double4 xi4 = __ldg(reinterpret_cast<const double4*>(x) + i);
  const double xtmp = xi4.x;
  const double ytmp = xi4.y;
  const double ztmp = xi4.z;

  const int type_i = __ldg(type + i);

  double fix = 0.0, fiy = 0.0, fiz = 0.0;

  const int base = i * maxneighs;
  const int nn   = __ldg(numneigh + i);

  // Neighbor loop (streaming neighbors; irregular gathers for x[j], type[j])
  #pragma unroll 2
  for (int k = 0; k < nn; ++k) {
    const int j = __ldg(neighbors + (base + k));

    const double4 xj4 = __ldg(reinterpret_cast<const double4*>(x) + j);
    const double delx = xtmp - xj4.x;
    const double dely = ytmp - xj4.y;
    const double delz = ztmp - xj4.z;

    // rsq = delx*delx + dely*dely + delz*delz using FMAs
    const double rsq = fma(delx, delx, fma(dely, dely, delz * delz));

    const int type_j  = __ldg(type + j);
    const int type_ij = type_i * ntypes + type_j;

    const double cut = __ldg(cutforcesq + type_ij);
    if (rsq < cut) {
      // Double precision: use reciprocal (compiler may lower to rcp+refine)
      const double sr2 = 1.0 / rsq;
      const double sr4 = sr2 * sr2;
      const double inv3 = sr4 * sr2; // sr2^3

      const double sig6 = __ldg(sigma6 + type_ij);
      const double eps  = __ldg(epsilon + type_ij);

      const double sr6 = inv3 * sig6;
      // force = 48 * sr6 * (sr6 - 0.5) * sr2 * eps
      const double t = sr6 * (sr6 - 0.5);
      const double force = (48.0 * eps) * (t * sr2);

      fix = fma(delx, force, fix);
      fiy = fma(dely, force, fiy);
      fiz = fma(delz, force, fiz);
    }
  }

  // Vectorized store for f[i] (assumes PAD==4 layout)
  double4 fi4;
  fi4.x = fix;
  fi4.y = fiy;
  fi4.z = fiz;
  fi4.w = 0.0;
  *reinterpret_cast<double4*>(f + (i << 2)) = fi4;
}

void lj_force_cpu(int nlocal, int ntypes, const double* x, double* f, const int* type,
    const int* neighbors, const int* numneigh, int maxneighs,
    const double* cutforcesq, const double* epsilon, const double* sigma6) {
    for (int i = 0; i < nlocal; i++) {
        double xtmp = x[i*4+0], ytmp = x[i*4+1], ztmp = x[i*4+2];
        int type_i = type[i];
        double fix = 0, fiy = 0, fiz = 0;
        for (int k = 0; k < numneigh[i]; k++) {
            int j = neighbors[i * maxneighs + k];
            double delx = xtmp - x[j*4+0];
            double dely = ytmp - x[j*4+1];
            double delz = ztmp - x[j*4+2];
            double rsq = delx*delx + dely*dely + delz*delz;
            int type_ij = type_i * ntypes + type[j];
            if (rsq < cutforcesq[type_ij]) {
                double sr2 = 1.0/rsq;
                double sr6 = sr2 * sr2 * sr2 * sigma6[type_ij];
                double force = 48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon[type_ij];
                fix += delx * force;
                fiy += dely * force;
                fiz += delz * force;
            }
        }
        f[i*4+0] = fix; f[i*4+1] = fiy; f[i*4+2] = fiz; f[i*4+3] = 0;
    }
}

int main() {
    int nlocal = 100000, ntypes = 1, maxneighs = 128;
    double *h_x, *h_f_cpu, *h_f_gpu, *h_cutforcesq, *h_epsilon, *h_sigma6;
    int *h_type, *h_neighbors, *h_numneigh;
    h_x = (double*)malloc(nlocal * 4 * sizeof(double));
    h_f_cpu = (double*)malloc(nlocal * 4 * sizeof(double));
    h_f_gpu = (double*)malloc(nlocal * 4 * sizeof(double));
    h_type = (int*)malloc(nlocal * sizeof(int));
    h_neighbors = (int*)malloc(nlocal * maxneighs * sizeof(int));
    h_numneigh = (int*)malloc(nlocal * sizeof(int));
    h_cutforcesq = (double*)malloc(ntypes * ntypes * sizeof(double));
    h_epsilon = (double*)malloc(ntypes * ntypes * sizeof(double));
    h_sigma6 = (double*)malloc(ntypes * ntypes * sizeof(double));
    
    srand(12345);
    for (int i = 0; i < nlocal; i++) {
        h_x[i*4+0] = (double)rand()/RAND_MAX * 100;
        h_x[i*4+1] = (double)rand()/RAND_MAX * 100;
        h_x[i*4+2] = (double)rand()/RAND_MAX * 100;
        h_x[i*4+3] = 0;
        h_type[i] = 0;
        h_numneigh[i] = 50 + rand() % 20;
        for (int k = 0; k < h_numneigh[i]; k++)
            h_neighbors[i * maxneighs + k] = rand() % nlocal;
    }
    h_cutforcesq[0] = 16.0; h_epsilon[0] = 1.0; h_sigma6[0] = 1.0;
    
    clock_t cpu_start = clock();
    for (int iter = 0; iter < 3; iter++) lj_force_cpu(nlocal, ntypes, h_x, h_f_cpu, h_type, h_neighbors, h_numneigh, maxneighs, h_cutforcesq, h_epsilon, h_sigma6);
    double cpu_ms = (double)(clock() - cpu_start) / CLOCKS_PER_SEC * 1000.0 / 3.0;
    
    double *d_x, *d_f, *d_cutforcesq, *d_epsilon, *d_sigma6;
    int *d_type, *d_neighbors, *d_numneigh;
    CHECK_CUDA(cudaMalloc(&d_x, nlocal * 4 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_f, nlocal * 4 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_type, nlocal * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_neighbors, nlocal * maxneighs * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_numneigh, nlocal * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_cutforcesq, ntypes * ntypes * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_epsilon, ntypes * ntypes * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_sigma6, ntypes * ntypes * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, nlocal * 4 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_type, h_type, nlocal * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_neighbors, h_neighbors, nlocal * maxneighs * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_numneigh, h_numneigh, nlocal * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cutforcesq, h_cutforcesq, sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_epsilon, h_epsilon, sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_sigma6, h_sigma6, sizeof(double), cudaMemcpyHostToDevice));
    
    int bs = 256, nb = (nlocal + bs - 1) / bs;
    lj_force_pipeline<<<nb, bs>>>(nlocal, ntypes, d_x, d_f, d_type, d_neighbors, d_numneigh, maxneighs, d_cutforcesq, d_epsilon, d_sigma6);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int iter = 0; iter < 10; iter++)
        lj_force_pipeline<<<nb, bs>>>(nlocal, ntypes, d_x, d_f, d_type, d_neighbors, d_numneigh, maxneighs, d_cutforcesq, d_epsilon, d_sigma6);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_ms; cudaEventElapsedTime(&gpu_ms, start, stop); gpu_ms /= 10;
    
    CHECK_CUDA(cudaMemcpy(h_f_gpu, d_f, nlocal * 4 * sizeof(double), cudaMemcpyDeviceToHost));
    double maxerr = 0;
    for (int i = 0; i < nlocal * 4; i++) { double e = fabs(h_f_cpu[i] - h_f_gpu[i]); if (e > maxerr) maxerr = e; }
    printf("BENCHMARK_RESULT:cpu_ms=%.4f,gpu_ms=%.4f,speedup=%.2f,error=%.2e\n", cpu_ms, gpu_ms, cpu_ms/gpu_ms, maxerr);
    return 0;
}
