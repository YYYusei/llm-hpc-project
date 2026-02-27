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