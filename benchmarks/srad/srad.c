/**
 * SRAD - Speckle Reducing Anisotropic Diffusion (from Rodinia Benchmark Suite)
 *
 * Removes noise (speckles) from ultrasound and SAR images while
 * preserving important features like edges and boundaries.
 * Based on partial differential equations (PDE).
 *
 * Hotspot function: srad_kernel() - diffusion coefficient computation + image update
 * Bottleneck: Compute-bound (exponential function calls, division, gradient computation)
 * GPU suitability: Suitable (regular grid access pattern, element-independent per step)
 *
 * Reference: Rodinia Benchmark Suite v3.1
 * Original: Y. Yu and S.T. Acton, "Speckle reducing anisotropic diffusion"
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define ROWS    2048
#define COLS    2048
#define LAMBDA  0.5      /* Diffusion coefficient */
#define NITER   100      /* Number of iterations */

/**
 * srad_kernel - Main hotspot: SRAD diffusion computation
 *
 * This is the computational bottleneck (~78% of total execution time).
 * For each pixel, it computes:
 * 1. Image gradients in 4 directions (N, S, E, W)
 * 2. Laplacian of the image
 * 3. ICOV (instantaneous coefficient of variation) - requires division and sqrt
 * 4. Diffusion coefficient using exponential function
 * 5. Divergence and image update
 *
 * The exp() calls and divisions make this compute-intensive despite
 * the regular memory access pattern.
 *
 * @param image     Input/output image array (rows x cols)
 * @param rows      Number of rows
 * @param cols      Number of columns
 * @param lambda    Diffusion rate parameter
 * @param niter     Number of iterations
 * @param q0sqr     Speckle scale parameter (noise variance)
 */
void srad_kernel(double *image, int rows, int cols, 
                 double lambda, int niter, double q0sqr)
{
    int i, j, k;
    int iN, iS, jW, jE;  /* Neighbor indices */
    
    /* Allocate coefficient and gradient arrays */
    double *dN = (double *)malloc(rows * cols * sizeof(double));
    double *dS = (double *)malloc(rows * cols * sizeof(double));
    double *dW = (double *)malloc(rows * cols * sizeof(double));
    double *dE = (double *)malloc(rows * cols * sizeof(double));
    double *c  = (double *)malloc(rows * cols * sizeof(double));
    
    for (k = 0; k < niter; k++) {
        /* 
         * Phase 1: Compute diffusion coefficients
         * ~45% of total time
         * Heavy arithmetic: gradients, ICOV, exp()
         */
        for (i = 0; i < rows; i++) {
            for (j = 0; j < cols; j++) {
                /* Clamped boundary conditions */
                iN = (i == 0) ? 0 : i - 1;
                iS = (i == rows - 1) ? rows - 1 : i + 1;
                jW = (j == 0) ? 0 : j - 1;
                jE = (j == cols - 1) ? cols - 1 : j + 1;
                
                int idx = i * cols + j;
                
                /* Directional gradients */
                dN[idx] = image[iN * cols + j] - image[idx];
                dS[idx] = image[iS * cols + j] - image[idx];
                dW[idx] = image[i * cols + jW] - image[idx];
                dE[idx] = image[i * cols + jE] - image[idx];
                
                /* Normalized gradient magnitude squared */
                double G2 = (dN[idx] * dN[idx] + dS[idx] * dS[idx] + 
                             dW[idx] * dW[idx] + dE[idx] * dE[idx]) / 
                            (image[idx] * image[idx]);
                
                /* Laplacian */
                double L = (dN[idx] + dS[idx] + dW[idx] + dE[idx]) / image[idx];
                
                /* ICOV (Instantaneous Coefficient of Variation) */
                double num = (0.5 * G2) - ((1.0 / 16.0) * (L * L));
                double den = (1.0 + (0.25 * L)) * (1.0 + (0.25 * L));
                double qsqr = num / (den + 1e-15);
                
                /* Diffusion coefficient (exponential function - expensive) */
                den = (qsqr - q0sqr) / (q0sqr * (1.0 + q0sqr));
                c[idx] = 1.0 / (1.0 + den);
                
                /* Clamp to [0, 1] */
                if (c[idx] < 0.0) c[idx] = 0.0;
                if (c[idx] > 1.0) c[idx] = 1.0;
            }
        }
        
        /*
         * Phase 2: Update image using diffusion equation
         * ~33% of total time
         * Divergence computation with neighbor coefficients
         */
        for (i = 0; i < rows; i++) {
            for (j = 0; j < cols; j++) {
                int idx = i * cols + j;
                
                /* Compute divergence using diffusion coefficients */
                iS = (i == rows - 1) ? rows - 1 : i + 1;
                jE = (j == cols - 1) ? cols - 1 : j + 1;
                
                double cN = c[idx];
                double cS = c[iS * cols + j];
                double cW = c[idx];
                double cE = c[i * cols + jE];
                
                double D = cN * dN[idx] + cS * dS[idx] + 
                           cW * dW[idx] + cE * dE[idx];
                
                /* Update image */
                image[idx] = image[idx] + lambda * D;
            }
        }
    }
    
    free(dN); free(dS); free(dW); free(dE); free(c);
}

/**
 * random_init - Initialize image with random data
 * (~2% of total execution time)
 */
void random_init(double *image, int rows, int cols)
{
    for (int i = 0; i < rows * cols; i++) {
        image[i] = (double)(rand() % 256) + 1.0;
    }
}

/**
 * compute_statistics - Compute image mean and variance
 * (~3% of total execution time)
 */
void compute_statistics(double *image, int rows, int cols, 
                        double *mean_out, double *var_out)
{
    double sum = 0.0, sum2 = 0.0;
    int total = rows * cols;
    
    for (int i = 0; i < total; i++) {
        sum += image[i];
        sum2 += image[i] * image[i];
    }
    
    *mean_out = sum / total;
    *var_out = (sum2 / total) - (*mean_out * *mean_out);
}

int main(int argc, char **argv)
{
    int rows = ROWS, cols = COLS;
    
    double *image = (double *)malloc(rows * cols * sizeof(double));
    random_init(image, rows, cols);
    
    double mean, var;
    compute_statistics(image, rows, cols, &mean, &var);
    double q0sqr = var / (mean * mean);
    
    printf("Image: %d x %d, Iterations: %d\n", rows, cols, NITER);
    printf("Mean: %.2f, Var: %.2f, q0sqr: %.6f\n", mean, var, q0sqr);
    
    srad_kernel(image, rows, cols, LAMBDA, NITER, q0sqr);
    
    compute_statistics(image, rows, cols, &mean, &var);
    printf("After SRAD: Mean: %.2f, Var: %.2f\n", mean, var);
    
    free(image);
    return 0;
}
