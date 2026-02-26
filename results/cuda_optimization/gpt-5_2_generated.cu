__global__ void optimized_kernel_gpt_5_2(int nlocal, int ntypes,
    const double* __restrict__ x, double* __restrict__ f, const int* __restrict__ type,
    const int* __restrict__ neighbors, const int* __restrict__ numneigh, int maxneighs,
    const double* __restrict__ cutforcesq, const double* __restrict__ epsilon,
    const double* __restrict__ sigma6)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nlocal) return;

    // Vectorized position/force access (double4)
    const double4* __restrict__ x4 = reinterpret_cast<const double4*>(x);
    double4* __restrict__ f4 = reinterpret_cast<double4*>(f);

    const double4 xi4 = x4[i];
    const double xtmp = xi4.x;
    const double ytmp = xi4.y;
    const double ztmp = xi4.z;

    const int type_i = ldg_int(type + i);

    double fix = 0.0, fiy = 0.0, fiz = 0.0;

    const int nneigh = ldg_int(numneigh + i);
    const int base = i * maxneighs;

    int k = 0;

    // Unroll by 4
    for (; k + 3 < nneigh; k += 4) {
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int j = ldg_int(neighbors + (base + k + u));

            const double4 xj4 = x4[j];
            const double delx = xtmp - xj4.x;
            const double dely = ytmp - xj4.y;
            const double delz = ztmp - xj4.z;

            const double rsq = fma(delx, delx, fma(dely, dely, delz * delz));

            const int type_j = ldg_int(type + j);
            const int type_ij = type_i * ntypes + type_j;

            const double cutsq = ldg_double(cutforcesq + type_ij);
            if (rsq < cutsq) {
                const double inv_rsq = 1.0 / rsq;
                const double inv_r4  = inv_rsq * inv_rsq;
                const double inv_r6  = inv_r4 * inv_rsq;

                const double s6  = ldg_double(sigma6 + type_ij);
                const double sr6 = inv_r6 * s6;

                const double eps = ldg_double(epsilon + type_ij);

                // force = 48 * eps * sr6 * (sr6 - 0.5) * inv_rsq
                const double t = sr6 - 0.5;
                const double force = (48.0 * eps) * (sr6 * t) * inv_rsq;

                fix = fma(delx, force, fix);
                fiy = fma(dely, force, fiy);
                fiz = fma(delz, force, fiz);
            }
        }
    }

    // Remainder
    for (; k < nneigh; ++k) {
        const int j = ldg_int(neighbors + (base + k));

        const double4 xj4 = x4[j];
        const double delx = xtmp - xj4.x;
        const double dely = ytmp - xj4.y;
        const double delz = ztmp - xj4.z;

        const double rsq = fma(delx, delx, fma(dely, dely, delz * delz));

        const int type_j = ldg_int(type + j);
        const int type_ij = type_i * ntypes + type_j;

        const double cutsq = ldg_double(cutforcesq + type_ij);
        if (rsq < cutsq) {
            const double inv_rsq = 1.0 / rsq;
            const double inv_r4  = inv_rsq * inv_rsq;
            const double inv_r6  = inv_r4 * inv_rsq;

            const double s6  = ldg_double(sigma6 + type_ij);
            const double sr6 = inv_r6 * s6;

            const double eps = ldg_double(epsilon + type_ij);

            const double t = sr6 - 0.5;
            const double force = (48.0 * eps) * (sr6 * t) * inv_rsq;

            fix = fma(delx, force, fix);
            fiy = fma(dely, force, fiy);
            fiz = fma(delz, force, fiz);
        }
    }

    // Store (keep w component unchanged if present)
    double4 fi4 = f4[i];
    fi4.x = fix;
    fi4.y = fiy;
    fi4.z = fiz;
    f4[i] = fi4;
}