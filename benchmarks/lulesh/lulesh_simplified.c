/**
 * LULESH - Livermore Unstructured Lagrangian Explicit Shock Hydrodynamics
 * (Simplified version of the LLNL proxy application)
 *
 * Simulates shock hydrodynamics on an unstructured hexahedral mesh.
 * Core computation involves stress tensor calculation, force accumulation,
 * and energy/velocity updates per element.
 *
 * Hotspot function: CalcHourglassControlForElems() + CalcFBHourglassForceForElems()
 * Secondary hotspot: CalcKinematicsForElems()
 * Bottleneck: Compute-bound (heavy FP arithmetic per element, gather/scatter pattern)
 * GPU suitability: Suitable with caveats (element-parallel but gather/scatter on nodes)
 *
 * Reference: LULESH 2.0 (https://github.com/LLNL/LULESH)
 * Original authors: Jeff Keasler, Richard Hornung (LLNL)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Domain size */
#define SIDE_LENGTH     30
#define NUM_ELEMS       (SIDE_LENGTH * SIDE_LENGTH * SIDE_LENGTH)
#define NUM_NODES       ((SIDE_LENGTH+1) * (SIDE_LENGTH+1) * (SIDE_LENGTH+1))
#define NODES_PER_ELEM  8

/* Material constants */
#define GAMMA_COEFF     (1.0/3.0)
#define HGCOEF          3.0
#define SS4O3           (4.0/3.0)

/**
 * CalcKinematicsForElems - Compute element kinematics
 *
 * Secondary hotspot (~25% of total execution time).
 * Computes volume derivatives, velocity gradients, and strain rates
 * for each element by gathering node coordinates and velocities.
 *
 * @param x, y, z          Node coordinates
 * @param xd, yd, zd       Node velocities
 * @param nodelist          Element-to-node connectivity (8 nodes per elem)
 * @param vol               Element volumes (output)
 * @param vdov              Volume derivative (output)
 * @param delv              Volume change (output)
 * @param numElem           Number of elements
 */
void CalcKinematicsForElems(const double *x, const double *y, const double *z,
                            const double *xd, const double *yd, const double *zd,
                            const int *nodelist,
                            double *vol, double *vdov, double *delv,
                            int numElem, double dt)
{
    int k, lnode;
    
    for (k = 0; k < numElem; k++) {
        double x_local[8], y_local[8], z_local[8];
        double xd_local[8], yd_local[8], zd_local[8];
        
        /* Gather node data for this element (indirect indexing) */
        const int *elemToNode = &nodelist[k * NODES_PER_ELEM];
        for (lnode = 0; lnode < 8; lnode++) {
            int gnode = elemToNode[lnode];
            x_local[lnode] = x[gnode];
            y_local[lnode] = y[gnode];
            z_local[lnode] = z[gnode];
            xd_local[lnode] = xd[gnode];
            yd_local[lnode] = yd[gnode];
            zd_local[lnode] = zd[gnode];
        }
        
        /* Compute volume via triple product of diagonals */
        double dvdx = (y_local[3] - y_local[0]) * (z_local[2] - z_local[5]) -
                      (y_local[2] - y_local[5]) * (z_local[3] - z_local[0]);
        double dvdy = (z_local[3] - z_local[0]) * (x_local[2] - x_local[5]) -
                      (z_local[2] - z_local[5]) * (x_local[3] - x_local[0]);
        double dvdz = (x_local[3] - x_local[0]) * (y_local[2] - y_local[5]) -
                      (x_local[2] - x_local[5]) * (y_local[3] - y_local[0]);
        
        double det = (x_local[1] - x_local[6]) * dvdx +
                     (y_local[1] - y_local[6]) * dvdy +
                     (z_local[1] - z_local[6]) * dvdz;
        
        /* Additional cross-product contributions */
        dvdx = (y_local[4] - y_local[1]) * (z_local[7] - z_local[2]) -
               (y_local[7] - y_local[2]) * (z_local[4] - z_local[1]);
        dvdy = (z_local[4] - z_local[1]) * (x_local[7] - x_local[2]) -
               (z_local[7] - z_local[2]) * (x_local[4] - x_local[1]);
        dvdz = (x_local[4] - x_local[1]) * (y_local[7] - y_local[2]) -
               (x_local[7] - x_local[2]) * (y_local[4] - y_local[1]);
        
        det += (x_local[0] - x_local[5]) * dvdx +
               (y_local[0] - y_local[5]) * dvdy +
               (z_local[0] - z_local[5]) * dvdz;
        
        dvdx = (y_local[5] - y_local[6]) * (z_local[0] - z_local[3]) -
               (y_local[0] - y_local[3]) * (z_local[5] - z_local[6]);
        dvdy = (z_local[5] - z_local[6]) * (x_local[0] - x_local[3]) -
               (z_local[0] - z_local[3]) * (x_local[5] - x_local[6]);
        dvdz = (x_local[5] - x_local[6]) * (y_local[0] - y_local[3]) -
               (x_local[0] - x_local[3]) * (y_local[5] - y_local[6]);
        
        det += (x_local[7] - x_local[4]) * dvdx +
               (y_local[7] - y_local[4]) * dvdy +
               (z_local[7] - z_local[4]) * dvdz;
        
        det /= 12.0;
        vol[k] = det;
        
        /* Compute velocity gradient (strain rate) */
        double d6[6];
        d6[0] = 0.5 * ((xd_local[0] + xd_local[1] + xd_local[2] + xd_local[3]) -
                        (xd_local[4] + xd_local[5] + xd_local[6] + xd_local[7]));
        d6[1] = 0.5 * ((yd_local[0] + yd_local[1] + yd_local[4] + yd_local[5]) -
                        (yd_local[2] + yd_local[3] + yd_local[6] + yd_local[7]));
        d6[2] = 0.5 * ((zd_local[0] + zd_local[3] + zd_local[4] + zd_local[7]) -
                        (zd_local[1] + zd_local[2] + zd_local[5] + zd_local[6]));
        d6[3] = 0.5 * ((xd_local[0] + xd_local[3] + xd_local[4] + xd_local[7]) -
                        (xd_local[1] + xd_local[2] + xd_local[5] + xd_local[6]));
        d6[4] = 0.5 * ((yd_local[0] + yd_local[1] + yd_local[4] + yd_local[5]) -
                        (yd_local[2] + yd_local[3] + yd_local[6] + yd_local[7]));
        d6[5] = 0.5 * ((zd_local[0] + zd_local[1] + zd_local[2] + zd_local[3]) -
                        (zd_local[4] + zd_local[5] + zd_local[6] + zd_local[7]));
        
        /* Volume derivative over volume */
        vdov[k] = (d6[0] + d6[1] + d6[2]) / det;
        delv[k] = det - vol[k];
    }
}

/**
 * CalcFBHourglassForceForElems - Anti-hourglass force computation
 *
 * PRIMARY HOTSPOT (~45% of total execution time).
 * Computes hourglass forces to prevent zero-energy mesh deformation modes.
 * Very arithmetic-intensive: 8 nodes x 3 directions x 4 hourglass modes.
 *
 * @param xd, yd, zd       Node velocities
 * @param nodelist          Element-to-node connectivity
 * @param ss, elemMass      Sound speed and element mass
 * @param fx, fy, fz        Node force arrays (output, accumulated via scatter)
 * @param numElem           Number of elements
 */
void CalcFBHourglassForceForElems(const double *xd, const double *yd, const double *zd,
                                   const int *nodelist,
                                   const double *ss, const double *elemMass,
                                   double *fx, double *fy, double *fz,
                                   int numElem)
{
    /* Hourglass basis vectors (constant) */
    static const double gamma[4][8] = {
        { 1.0,  1.0, -1.0, -1.0, -1.0, -1.0,  1.0,  1.0},
        { 1.0, -1.0, -1.0,  1.0, -1.0,  1.0,  1.0, -1.0},
        { 1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0},
        {-1.0,  1.0, -1.0,  1.0,  1.0, -1.0,  1.0, -1.0}
    };
    
    double coefficient = HGCOEF;
    
    for (int i = 0; i < numElem; i++) {
        const int *elemToNode = &nodelist[i * NODES_PER_ELEM];
        double xd_local[8], yd_local[8], zd_local[8];
        double hgfx[8], hgfy[8], hgfz[8];
        
        /* Initialize hourglass forces to zero */
        for (int n = 0; n < 8; n++) {
            hgfx[n] = hgfy[n] = hgfz[n] = 0.0;
        }
        
        /* Gather velocities */
        for (int n = 0; n < 8; n++) {
            int gnode = elemToNode[n];
            xd_local[n] = xd[gnode];
            yd_local[n] = yd[gnode];
            zd_local[n] = zd[gnode];
        }
        
        double volume = elemMass[i] / ss[i];
        double volinv = 1.0 / (volume + 1e-20);
        double hourmodx, hourmody, hourmodz;
        
        /*
         * HOTSPOT: Inner loop over 4 hourglass modes
         * Each mode: dot product (8 mults + 7 adds) x 3 directions,
         * then scatter to 8 nodes x 3 directions = 192 FP ops per element
         * Total: ~768 FP ops per element for all 4 modes
         */
        for (int mode = 0; mode < 4; mode++) {
            /* Project velocities onto hourglass mode */
            hourmodx = 0.0;
            hourmody = 0.0;
            hourmodz = 0.0;
            
            for (int n = 0; n < 8; n++) {
                hourmodx += xd_local[n] * gamma[mode][n];
                hourmody += yd_local[n] * gamma[mode][n];
                hourmodz += zd_local[n] * gamma[mode][n];
            }
            
            /* Scale by coefficient and inverse volume */
            double hgcoef = coefficient * ss[i] * elemMass[i] * volinv;
            
            /* Accumulate hourglass forces for each node */
            for (int n = 0; n < 8; n++) {
                hgfx[n] -= hgcoef * hourmodx * gamma[mode][n];
                hgfy[n] -= hgcoef * hourmody * gamma[mode][n];
                hgfz[n] -= hgcoef * hourmodz * gamma[mode][n];
            }
        }
        
        /* Scatter forces to global node arrays (race condition in parallel!) */
        for (int n = 0; n < 8; n++) {
            int gnode = elemToNode[n];
            fx[gnode] += hgfx[n];
            fy[gnode] += hgfy[n];
            fz[gnode] += hgfz[n];
        }
    }
}

/**
 * CalcPressureForElems - Equation of state pressure calculation
 * (~10% of total execution time)
 */
void CalcPressureForElems(double *p, double *compression, double *vnew,
                          double pmin, double p_cut, int numElem)
{
    for (int i = 0; i < numElem; i++) {
        double c1s = 2.0 / 3.0;
        double bvc = c1s * (compression[i] + 1.0);
        double pbvc = c1s;
        
        p[i] = bvc * vnew[i];
        if (fabs(p[i]) < p_cut) p[i] = 0.0;
        if (vnew[i] >= 1.0) p[i] = 0.0;
        if (p[i] < pmin) p[i] = pmin;
    }
}

int main(int argc, char **argv)
{
    printf("LULESH Simplified: %d elements, %d nodes\n", NUM_ELEMS, NUM_NODES);
    
    /* Allocate arrays */
    double *x  = (double *)calloc(NUM_NODES, sizeof(double));
    double *y  = (double *)calloc(NUM_NODES, sizeof(double));
    double *z  = (double *)calloc(NUM_NODES, sizeof(double));
    double *xd = (double *)calloc(NUM_NODES, sizeof(double));
    double *yd = (double *)calloc(NUM_NODES, sizeof(double));
    double *zd = (double *)calloc(NUM_NODES, sizeof(double));
    double *fx = (double *)calloc(NUM_NODES, sizeof(double));
    double *fy = (double *)calloc(NUM_NODES, sizeof(double));
    double *fz = (double *)calloc(NUM_NODES, sizeof(double));
    
    int *nodelist   = (int *)malloc(NUM_ELEMS * NODES_PER_ELEM * sizeof(int));
    double *vol     = (double *)malloc(NUM_ELEMS * sizeof(double));
    double *vdov    = (double *)malloc(NUM_ELEMS * sizeof(double));
    double *delv    = (double *)malloc(NUM_ELEMS * sizeof(double));
    double *ss      = (double *)malloc(NUM_ELEMS * sizeof(double));
    double *elemMass = (double *)malloc(NUM_ELEMS * sizeof(double));
    
    /* Initialize mesh (regular hexahedral grid) */
    int nidx = 0;
    int s1 = SIDE_LENGTH + 1;
    for (int k = 0; k <= SIDE_LENGTH; k++)
        for (int j = 0; j <= SIDE_LENGTH; j++)
            for (int i = 0; i <= SIDE_LENGTH; i++) {
                x[nidx] = 1.125 * (double)i / SIDE_LENGTH;
                y[nidx] = 1.125 * (double)j / SIDE_LENGTH;
                z[nidx] = 1.125 * (double)k / SIDE_LENGTH;
                xd[nidx] = 0.0;
                yd[nidx] = 0.0;
                zd[nidx] = -1.0e-2 * z[nidx];
                nidx++;
            }
    
    /* Build element connectivity */
    int eidx = 0;
    for (int k = 0; k < SIDE_LENGTH; k++)
        for (int j = 0; j < SIDE_LENGTH; j++)
            for (int i = 0; i < SIDE_LENGTH; i++) {
                int base = k * s1 * s1 + j * s1 + i;
                nodelist[eidx * 8 + 0] = base;
                nodelist[eidx * 8 + 1] = base + 1;
                nodelist[eidx * 8 + 2] = base + s1 + 1;
                nodelist[eidx * 8 + 3] = base + s1;
                nodelist[eidx * 8 + 4] = base + s1 * s1;
                nodelist[eidx * 8 + 5] = base + s1 * s1 + 1;
                nodelist[eidx * 8 + 6] = base + s1 * s1 + s1 + 1;
                nodelist[eidx * 8 + 7] = base + s1 * s1 + s1;
                ss[eidx] = 1.0;
                elemMass[eidx] = 1.0;
                eidx++;
            }
    
    printf("Running kinematics...\n");
    CalcKinematicsForElems(x, y, z, xd, yd, zd, nodelist,
                           vol, vdov, delv, NUM_ELEMS, 1.0e-5);
    
    printf("Running hourglass forces...\n");
    CalcFBHourglassForceForElems(xd, yd, zd, nodelist, ss, elemMass,
                                 fx, fy, fz, NUM_ELEMS);
    
    printf("Complete. Max force: %.6e\n", fx[0]);
    
    free(x); free(y); free(z);
    free(xd); free(yd); free(zd);
    free(fx); free(fy); free(fz);
    free(nodelist); free(vol); free(vdov); free(delv);
    free(ss); free(elemMass);
    
    return 0;
}
