```cuda
#define PAD 4

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
    f[i * PAD + 3] = 0.0;
}
```