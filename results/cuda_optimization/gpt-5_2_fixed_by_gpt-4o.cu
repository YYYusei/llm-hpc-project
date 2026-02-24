__global__ void optimized_kernel_gpt_5_2(int nlocal, int ntypes,
    const double* __restrict__ x, double* __restrict__ f, const int* __restrict__ type,
    const int* __restrict__ neighbors, const int* __restrict__ numneigh, int maxneighs,
    const double* __restrict__ cutforcesq, const double* __restrict__ epsilon,
    const double* __restrict__ sigma6)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nlocal) return;

    // Vectorized loads/stores for x and f (assumes x/f are 16-byte aligned as in typical LAMMPS layouts)
    const double4* __restrict__ x4 = reinterpret_cast<const double4*>(x);
    double4* __restrict__ f4 = reinterpret_cast<double4*>(f);

    // Directly access the double4 without using __ldg for double4
    const double4 xi4 = x4[i];
    const double xtmp = xi4.x;
    const double ytmp = xi4.y;
    const double ztmp = xi4.z;

    const int type_i = __ldg(&type[i]);
    const int nneigh = __ldg(&numneigh[i]);
    const int base = i * maxneighs;

    double fix = 0.0, fiy = 0.0, fiz = 0.0;

    // Unroll by 4; handle tail
    int k = 0;
#pragma unroll 4
    for (; k + 3 < nneigh; k += 4) {
        // ---- neighbor 0
        int j0 = __ldg(&neighbors[base + k + 0]);
        double4 xj0 = __ldg(&x4[j0]);
        double dx0 = xtmp - xj0.x;
        double dy0 = ytmp - xj0.y;
        double dz0 = ztmp - xj0.z;
        double rsq0 = fma(dx0, dx0, fma(dy0, dy0, dz0 * dz0));
        int tij0 = type_i * ntypes + __ldg(&type[j0]);
        double cut0 = __ldg(&cutforcesq[tij0]);
        if (rsq0 < cut0) {
            double inv0 = 1.0 / rsq0;
            double inv2_0 = inv0 * inv0;
            double inv6_0 = inv2_0 * inv2_0 * inv2_0;
            double sr6_0 = inv6_0 * __ldg(&sigma6[tij0]);
            double eps0 = __ldg(&epsilon[tij0]);
            double force0 = (48.0 * eps0) * (sr6_0 * (sr6_0 - 0.5) * inv0);
            fix = fma(dx0, force0, fix);
            fiy = fma(dy0, force0, fiy);
            fiz = fma(dz0, force0, fiz);
        }

        // ---- neighbor 1
        int j1 = __ldg(&neighbors[base + k + 1]);
        double4 xj1 = __ldg(&x4[j1]);
        double dx1 = xtmp - xj1.x;
        double dy1 = ytmp - xj1.y;
        double dz1 = ztmp - xj1.z;
        double rsq1 = fma(dx1, dx1, fma(dy1, dy1, dz1 * dz1));
        int tij1 = type_i * ntypes + __ldg(&type[j1]);
        double cut1 = __ldg(&cutforcesq[tij1]);
        if (rsq1 < cut1) {
            double inv1 = 1.0 / rsq1;
            double inv2_1 = inv1 * inv1;
            double inv6_1 = inv2_1 * inv2_1 * inv2_1;
            double sr6_1 = inv6_1 * __ldg(&sigma6[tij1]);
            double eps1 = __ldg(&epsilon[tij1]);
            double force1 = (48.0 * eps1) * (sr6_1 * (sr6_1 - 0.5) * inv1);
            fix = fma(dx1, force1, fix);
            fiy = fma(dy1, force1, fiy);
            fiz = fma(dz1, force1, fiz);
        }

        // ---- neighbor 2
        int j2 = __ldg(&neighbors[base + k + 2]);
        double4 xj2 = __ldg(&x4[j2]);
        double dx2 = xtmp - xj2.x;
        double dy2 = ytmp - xj2.y;
        double dz2 = ztmp - xj2.z;
        double rsq2 = fma(dx2, dx2, fma(dy2, dy2, dz2 * dz2));
        int tij2 = type_i * ntypes + __ldg(&type[j2]);
        double cut2 = __ldg(&cutforcesq[tij2]);
        if (rsq2 < cut2) {
            double inv2 = 1.0 / rsq2;
            double inv2_2 = inv2 * inv2;
            double inv6_2 = inv2_2 * inv2_2 * inv2_2;
            double sr6_2 = inv6_2 * __ldg(&sigma6[tij2]);
            double eps2 = __ldg(&epsilon[tij2]);
            double force2 = (48.0 * eps2) * (sr6_2 * (sr6_2 - 0.5) * inv2);
            fix = fma(dx2, force2, fix);
            fiy = fma(dy2, force2, fiy);
            fiz = fma(dz2, force2, fiz);
        }

        // ---- neighbor 3
        int j3 = __ldg(&neighbors[base + k + 3]);
        double4 xj3 = __ldg(&x4[j3]);
        double dx3 = xtmp - xj3.x;
        double dy3 = ytmp - xj3.y;
        double dz3 = ztmp - xj3.z;
        double rsq3 = fma(dx3, dx3, fma(dy3, dy3, dz3 * dz3));
        int tij3 = type_i * ntypes + __ldg(&type[j3]);
        double cut3 = __ldg(&cutforcesq[tij3]);
        if (rsq3 < cut3) {
            double inv3 = 1.0 / rsq3;
            double inv2_3 = inv3 * inv3;
            double inv6_3 = inv2_3 * inv2_3 * inv2_3;
            double sr6_3 = inv6_3 * __ldg(&sigma6[tij3]);
            double eps3 = __ldg(&epsilon[tij3]);
            double force3 = (48.0 * eps3) * (sr6_3 * (sr6_3 - 0.5) * inv3);
            fix = fma(dx3, force3, fix);
            fiy = fma(dy3, force3, fiy);
            fiz = fma(dz3, force3, fiz);
        }
    }

    for (; k < nneigh; ++k) {
        const int j = __ldg(&neighbors[base + k]);
        const double4 xj4 = __ldg(&x4[j]);

        const double dx = xtmp - xj4.x;
        const double dy = ytmp - xj4.y;
        const double dz = ztmp - xj4.z;

        const double rsq = fma(dx, dx, fma(dy, dy, dz * dz));
        const int tij = type_i * ntypes + __ldg(&type[j]);

        if (rsq < __ldg(&cutforcesq[tij])) {
            const double inv = 1.0 / rsq;
            const double inv2 = inv * inv;
            const double inv6 = inv2 * inv2 * inv2;
            const double sr6 = inv6 * __ldg(&sigma6[tij]);
            const double eps = __ldg(&epsilon[tij]);
            const double force = (48.0 * eps) * (sr6 * (sr6 - 0.5) * inv);

            fix = fma(dx, force, fix);
            fiy = fma(dy, force, fiy);
            fiz = fma(dz, force, fiz);
        }
    }

    // Preserve f.w (if used elsewhere) by reading then writing back with updated xyz
    double4 fi4 = f4[i];
    fi4.x = fix;
    fi4.y = fiy;
    fi4.z = fiz;
    f4[i] = fi4;
}