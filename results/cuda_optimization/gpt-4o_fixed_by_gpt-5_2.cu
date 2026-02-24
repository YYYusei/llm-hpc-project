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