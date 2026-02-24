// Optimized kernel code for Lennard-Jones force calculation

__global__ void compute_fullneigh_kernel_optimized(int nlocal, int ntypes,
    const double* __restrict__ x, double* __restrict__ f, const int* __restrict__ type,
    const int* __restrict__ neighbors, const int* __restrict__ numneigh, int maxneighs,
    const double* __restrict__ cutforcesq, const double* __restrict__ epsilon,
    const double* __restrict__ sigma6) {

    // Thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nlocal) return;

    // Shared memory for caching neighbor data
    extern __shared__ double shared_data[];
    double* shared_x = shared_data;
    int* shared_type = (int*)&shared_x[blockDim.x * 4];

    // Load current atom's position and type into registers
    double xtmp = x[i * 4 + 0];
    double ytmp = x[i * 4 + 1];
    double ztmp = x[i * 4 + 2];
    int type_i = type[i];

    // Initialize force accumulators
    double fix = 0.0, fiy = 0.0, fiz = 0.0;

    // Number of neighbors and base index for neighbors
    int nneigh = numneigh[i];
    int base = i * maxneighs;

    // Load neighbor data into shared memory
    for (int k = threadIdx.x; k < nneigh; k += blockDim.x) {
        int j = neighbors[base + k];
        shared_x[k * 4 + 0] = x[j * 4 + 0];
        shared_x[k * 4 + 1] = x[j * 4 + 1];
        shared_x[k * 4 + 2] = x[j * 4 + 2];
        shared_type[k] = type[j];
    }
    __syncthreads();

    // Loop over neighbors
    for (int k = 0; k < nneigh; k++) {
        // Load neighbor position and type from shared memory
        double delx = xtmp - shared_x[k * 4 + 0];
        double dely = ytmp - shared_x[k * 4 + 1];
        double delz = ztmp - shared_x[k * 4 + 2];
        int type_j = shared_type[k];

        // Calculate squared distance
        double rsq = delx * delx + dely * dely + delz * delz;
        int type_ij = type_i * ntypes + type_j;

        // Check cutoff and compute force if within range
        if (rsq < cutforcesq[type_ij]) {
            double sr2 = 1.0 / rsq;
            double sr6 = sr2 * sr2 * sr2 * sigma6[type_ij];
            double force = 48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon[type_ij];
            fix += delx * force;
            fiy += dely * force;
            fiz += delz * force;
        }
    }

    // Store results back to global memory
    f[i * 4 + 0] = fix;
    f[i * 4 + 1] = fiy;
    f[i * 4 + 2] = fiz;
}
