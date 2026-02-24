__global__ void optimized_kernel_gpt_4o(int nlocal, int ntypes,
    const double* __restrict__ x, double* __restrict__ f, const int* __restrict__ type,
    const int* __restrict__ neighbors, const int* __restrict__ numneigh, int maxneighs,
    const double* __restrict__ cutforcesq, const double* __restrict__ epsilon,
    const double* __restrict__ sigma6) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nlocal) return;

    // Use double4 for coalesced memory access
    double4 pos_i = reinterpret_cast<const double4*>(x)[i];
    double xtmp = pos_i.x, ytmp = pos_i.y, ztmp = pos_i.z;
    int type_i = ldg_int(&type[i]);

    double fix = 0.0, fiy = 0.0, fiz = 0.0;
    int nneigh = ldg_int(&numneigh[i]);
    int base = i * maxneighs;

    for (int k = 0; k < nneigh; k++) {
        int j = ldg_int(&neighbors[base + k]);

        // Use double4 for coalesced memory access
        double4 pos_j = reinterpret_cast<const double4*>(x)[j];
        double delx = xtmp - pos_j.x;
        double dely = ytmp - pos_j.y;
        double delz = ztmp - pos_j.z;

        double rsq = fma(delx, delx, fma(dely, dely, delz * delz));
        int type_ij = type_i * ntypes + ldg_int(&type[j]);

        double cutforcesq_ij = __ldg(&cutforcesq[type_ij]);
        if (rsq < cutforcesq_ij) {
            double sr2 = 1.0 / rsq;
            double sr6 = sr2 * sr2 * sr2 * __ldg(&sigma6[type_ij]);
            double epsilon_ij = __ldg(&epsilon[type_ij]);
            double force = 48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon_ij;

            fix = fma(delx, force, fix);
            fiy = fma(dely, force, fiy);
            fiz = fma(delz, force, fiz);
        }
    }

    // Use double4 for coalesced memory access
    reinterpret_cast<double4*>(f)[i] = make_double4(fix, fiy, fiz, 0.0);
}