// Optimized Lennard-Jones full-neighbor force kernel
// Target: SM 8.6 (RTX 3060). Compilable CUDA C++.
//
// Key changes vs baseline:
//  1) Memory coalescing: use double4 loads/stores for x/f (AoS-of-4 already present)
//  2) Shared memory: stage per-block i-particle positions/types to reduce redundant global loads
//  3) Read-only cache: use __ldg() for neighbor data and LJ tables (works on cc>=3.5; on newer GPUs it maps to LDG path)
//  4) Loop unrolling: unroll neighbor loop by 4 (typical 30-50 neighbors)
//  5) Register optimization: keep accumulators in registers; avoid extra temporaries; use FMA where possible
//  6) Warp-level optimization: warp-uniform early exit for empty neighbor lists (minor but cheap)
//
// Notes:
//  - This kernel computes forces on i only (no Newton's 3rd law update to j), matching baseline.
//  - Assumes x and f are 16-byte aligned (typical for cudaMalloc). If not guaranteed, it still works but may be slightly slower.
//  - neighbors is assumed to be a dense list with stride maxneighs per i (as in baseline).

#include <cuda_runtime.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

// Read-only load helper
__device__ __forceinline__ int ldg_int(const int* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}
__device__ __forceinline__ double ldg_double(const double* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

// Optional: tune block size for your occupancy/regs; 128 or 256 are typical sweet spots on Ampere.
#ifndef LJ_BLOCK_SIZE
#define LJ_BLOCK_SIZE 128
#endif

extern "C" __global__
void compute_fullneigh_kernel_opt(
    int nlocal, int ntypes,
    const double* __restrict__ x,          // length >= 4*nlocal (and includes ghost atoms as needed)
    double* __restrict__ f,                // length >= 4*nlocal
    const int* __restrict__ type,          // length >= nlocal (and includes ghost atoms as needed)
    const int* __restrict__ neighbors,     // length >= nlocal*maxneighs
    const int* __restrict__ numneigh,      // length >= nlocal
    int maxneighs,
    const double* __restrict__ cutforcesq, // length ntypes*ntypes
    const double* __restrict__ epsilon,    // length ntypes*ntypes
    const double* __restrict__ sigma6      // length ntypes*ntypes
) {
    // ---------------------------
    // [Optimization 1: Shared Memory]
    // Stage i-particle data for the block: position (double4) and type (int).
    // This avoids reloading x/type for i from global multiple times (and helps instruction scheduling).
    // ---------------------------
    __shared__ double4 sh_xi[LJ_BLOCK_SIZE];
    __shared__ int    sh_ti[LJ_BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int i   = blockIdx.x * blockDim.x + tid;

    // Guard for out-of-range threads
    if (i < nlocal) {
        // ---------------------------
        // [Optimization 2: Memory Coalescing]
        // Load x as double4 (16B-aligned vector load). x is stored as x[i*4 + {0,1,2,3}].
        // ---------------------------
        const double4 xi = reinterpret_cast<const double4*>(x)[i];
        sh_xi[tid] = xi;
        sh_ti[tid] = ldg_int(&type[i]);
    }
    __syncthreads();

    if (i >= nlocal) return;

    const double4 xi = sh_xi[tid];
    const int type_i = sh_ti[tid];

    // Accumulators in registers
    double fix = 0.0, fiy = 0.0, fiz = 0.0;

    // Load neighbor count (read-only path)
    const int nneigh = ldg_int(&numneigh[i]);

    // ---------------------------
    // [Optimization 6: Warp-level Optimization]
    // If an entire warp has nneigh==0, return early (saves some overhead in sparse cases).
    // ---------------------------
    const unsigned mask = 0xFFFFFFFFu;
#if __CUDA_ARCH__ >= 700
    if (__all_sync(mask, nneigh == 0)) {
        // Still need to write zeros for valid i
        reinterpret_cast<double4*>(f)[i] = make_double4(0.0, 0.0, 0.0, 0.0);
        return;
    }
#endif

    const int base = i * maxneighs;

    // ---------------------------
    // [Optimization 3: Read-only cache]
    // Use __ldg for neighbors and LJ tables (cutforcesq/epsilon/sigma6).
    // ---------------------------

    // ---------------------------
    // [Optimization 4: Loop Unrolling]
    // Unroll by 4 to reduce loop overhead and increase ILP.
    // ---------------------------
    int k = 0;
#pragma unroll 1
    for (; k + 3 < nneigh; k += 4) {
        // Load 4 neighbor indices
        const int j0 = ldg_int(&neighbors[base + k + 0]);
        const int j1 = ldg_int(&neighbors[base + k + 1]);
        const int j2 = ldg_int(&neighbors[base + k + 2]);
        const int j3 = ldg_int(&neighbors[base + k + 3]);

        // Coalesced-ish vector loads for x[j] (random access, but vectorized reduces instruction count)
        const double4 xj0 = reinterpret_cast<const double4*>(x)[j0];
        const double4 xj1 = reinterpret_cast<const double4*>(x)[j1];
        const double4 xj2 = reinterpret_cast<const double4*>(x)[j2];
        const double4 xj3 = reinterpret_cast<const double4*>(x)[j3];

        // Types
        const int tj0 = ldg_int(&type[j0]);
        const int tj1 = ldg_int(&type[j1]);
        const int tj2 = ldg_int(&type[j2]);
        const int tj3 = ldg_int(&type[j3]);

        // Compute 4 interactions
        // ---------------------------
        // [Optimization 5: Register Optimization]
        // Keep temporaries minimal; use fused multiply-add where compiler can.
        // ---------------------------

        // Interaction 0
        {
            const double delx = xi.x - xj0.x;
            const double dely = xi.y - xj0.y;
            const double delz = xi.z - xj0.z;
            const double rsq  = delx * delx + dely * dely + delz * delz;

            const int type_ij = type_i * ntypes + tj0;
            const double cutsq = ldg_double(&cutforcesq[type_ij]);
            if (rsq < cutsq) {
                const double invrsq = 1.0 / rsq;
                const double s6     = ldg_double(&sigma6[type_ij]);
                const double eps    = ldg_double(&epsilon[type_ij]);

                const double sr2 = invrsq;
                const double sr6 = (sr2 * sr2 * sr2) * s6;
                const double force = (48.0 * eps) * sr6 * (sr6 - 0.5) * sr2;

                fix = fma(delx, force, fix);
                fiy = fma(dely, force, fiy);
                fiz = fma(delz, force, fiz);
            }
        }

        // Interaction 1
        {
            const double delx = xi.x - xj1.x;
            const double dely = xi.y - xj1.y;
            const double delz = xi.z - xj1.z;
            const double rsq  = delx * delx + dely * dely + delz * delz;

            const int type_ij = type_i * ntypes + tj1;
            const double cutsq = ldg_double(&cutforcesq[type_ij]);
            if (rsq < cutsq) {
                const double invrsq = 1.0 / rsq;
                const double s6     = ldg_double(&sigma6[type_ij]);
                const double eps    = ldg_double(&epsilon[type_ij]);

                const double sr2 = invrsq;
                const double sr6 = (sr2 * sr2 * sr2) * s6;
                const double force = (48.0 * eps) * sr6 * (sr6 - 0.5) * sr2;

                fix = fma(delx, force, fix);
                fiy = fma(dely, force, fiy);
                fiz = fma(delz, force, fiz);
            }
        }

        // Interaction 2
        {
            const double delx = xi.x - xj2.x;
            const double dely = xi.y - xj2.y;
            const double delz = xi.z - xj2.z;
            const double rsq  = delx * delx + dely * dely + delz * delz;

            const int type_ij = type_i * ntypes + tj2;
            const double cutsq = ldg_double(&cutforcesq[type_ij]);
            if (rsq < cutsq) {
                const double invrsq = 1.0 / rsq;
                const double s6     = ldg_double(&sigma6[type_ij]);
                const double eps    = ldg_double(&epsilon[type_ij]);

                const double sr2 = invrsq;
                const double sr6 = (sr2 * sr2 * sr2) * s6;
                const double force = (48.0 * eps) * sr6 * (sr6 - 0.5) * sr2;

                fix = fma(delx, force, fix);
                fiy = fma(dely, force, fiy);
                fiz = fma(delz, force, fiz);
            }
        }

        // Interaction 3
        {
            const double delx = xi.x - xj3.x;
            const double dely = xi.y - xj3.y;
            const double delz = xi.z - xj3.z;
            const double rsq  = delx * delx + dely * dely + delz * delz;

            const int type_ij = type_i * ntypes + tj3;
            const double cutsq = ldg_double(&cutforcesq[type_ij]);
            if (rsq < cutsq) {
                const double invrsq = 1.0 / rsq;
                const double s6     = ldg_double(&sigma6[type_ij]);
                const double eps    = ldg_double(&epsilon[type_ij]);

                const double sr2 = invrsq;
                const double sr6 = (sr2 * sr2 * sr2) * s6;
                const double force = (48.0 * eps) * sr6 * (sr6 - 0.5) * sr2;

                fix = fma(delx, force, fix);
                fiy = fma(dely, force, fiy);
                fiz = fma(delz, force, fiz);
            }
        }
    }

    // Remainder loop
#pragma unroll 1
    for (; k < nneigh; k++) {
        const int j = ldg_int(&neighbors[base + k]);
        const double4 xj = reinterpret_cast<const double4*>(x)[j];

        const double delx = xi.x - xj.x;
        const double dely = xi.y - xj.y;
        const double delz = xi.z - xj.z;
        const double rsq  = delx * delx + dely * dely + delz * delz;

        const int tj = ldg_int(&type[j]);
        const int type_ij = type_i * ntypes + tj;

        const double cutsq = ldg_double(&cutforcesq[type_ij]);
        if (rsq < cutsq) {
            const double invrsq = 1.0 / rsq;
            const double s6     = ldg_double(&sigma6[type_ij]);
            const double eps    = ldg_double(&epsilon[type_ij]);

            const double sr2 = invrsq;
            const double sr6 = (sr2 * sr2 * sr2) * s6;
            const double force = (48.0 * eps) * sr6 * (sr6 - 0.5) * sr2;

            fix = fma(delx, force, fix);
            fiy = fma(dely, force, fiy);
            fiz = fma(delz, force, fiz);
        }
    }

    // ---------------------------
    // [Optimization 2: Memory Coalescing]
    // Store f as double4 (vector store). Keep w component as 0.
    // ---------------------------
    reinterpret_cast<double4*>(f)[i] = make_double4(fix, fiy, fiz, 0.0);
}
