/**
 * Jacobi-2D - Iterative Jacobi Stencil (from PolyBench Benchmark Suite)
 *
 * Solves the 2D Laplace equation using Jacobi iterative method.
 * Classic example of a memory-bound stencil computation with
 * no data dependencies within each iteration (unlike Gauss-Seidel).
 *
 * Hotspot function: jacobi_kernel()
 * Bottleneck: Memory-bound (5-point stencil, high memory traffic, low arithmetic intensity)
 * GPU suitability: Highly suitable (no data dependency within iteration, regular access)
 *
 * The key difference from Gauss-Seidel (like HPCG SYMGS) is that Jacobi
 * reads from one array and writes to another, making it trivially parallel
 * within each iteration.
 *
 * Reference: PolyBench/C 4.2
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define N       4096    /* Grid dimension */
#define TSTEPS  500     /* Number of time steps */

/**
 * jacobi_kernel - Main hotspot: 2D Jacobi stencil iteration
 *
 * PRIMARY HOTSPOT (~90% of total execution time).
 * Each grid point is updated as the average of its 4 neighbors plus itself.
 *
 * Memory-bound because:
 * - 5 loads + 1 store per grid point = 48 bytes of memory traffic
 * - Only 4 additions + 1 multiply (0.2) = 5 FLOPs
 * - Arithmetic intensity: 5/48 ≈ 0.10 FLOPs/byte (very low)
 * - Working set (2 * N * N * 8 bytes) exceeds cache for large N
 *
 * No data dependency within each sweep → trivially parallel, unlike Gauss-Seidel
 * which updates in-place and has read-after-write dependencies.
 *
 * @param tsteps  Number of iterations
 * @param n       Grid dimension
 * @param A       Input/output grid
 * @param B       Temporary grid (ping-pong buffer)
 */
void jacobi_kernel(int tsteps, int n, double *A, double *B)
{
    int t, i, j;
    
    for (t = 0; t < tsteps; t++) {
        /*
         * HOTSPOT: Forward sweep - A → B
         * 5-point stencil: center + north + south + east + west
         * No dependencies between grid points in this sweep
         */
        for (i = 1; i < n - 1; i++) {
            for (j = 1; j < n - 1; j++) {
                B[i * n + j] = 0.2 * (A[i * n + j] + 
                                      A[i * n + j - 1] +    /* west */
                                      A[i * n + j + 1] +    /* east */
                                      A[(i + 1) * n + j] +  /* south */
                                      A[(i - 1) * n + j]);  /* north */
            }
        }
        
        /*
         * HOTSPOT: Backward sweep - B → A
         * Same stencil operation, swapping source and destination
         */
        for (i = 1; i < n - 1; i++) {
            for (j = 1; j < n - 1; j++) {
                A[i * n + j] = 0.2 * (B[i * n + j] + 
                                      B[i * n + j - 1] +
                                      B[i * n + j + 1] +
                                      B[(i + 1) * n + j] +
                                      B[(i - 1) * n + j]);
            }
        }
    }
}

/**
 * init_array - Initialize grid with boundary conditions
 * (~1% of total execution time)
 */
void init_array(int n, double *A, double *B)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] = ((double)(i * (j + 2) + 2)) / n;
            B[i * n + j] = ((double)(i * (j + 3) + 3)) / n;
        }
    }
}

/**
 * compute_residual - Compute L2 residual of the Laplacian
 * (~3% of total execution time)
 */
double compute_residual(int n, const double *A)
{
    double residual = 0.0;
    
    for (int i = 1; i < n - 1; i++) {
        for (int j = 1; j < n - 1; j++) {
            double laplacian = A[(i - 1) * n + j] + A[(i + 1) * n + j] +
                               A[i * n + (j - 1)] + A[i * n + (j + 1)] -
                               4.0 * A[i * n + j];
            residual += laplacian * laplacian;
        }
    }
    
    return sqrt(residual / ((n - 2) * (n - 2)));
}

/**
 * compute_checksum - Simple verification checksum
 * (~1% of total execution time)
 */
double compute_checksum(int n, const double *A)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            sum += A[i * n + j];
        }
    }
    return sum;
}

int main(int argc, char **argv)
{
    int n = N;
    int tsteps = TSTEPS;
    
    printf("Jacobi-2D: Grid %d x %d, Iterations: %d\n", n, n, tsteps);
    printf("Working set: %.1f MB (2 grids)\n", 
           2.0 * n * n * sizeof(double) / (1024.0 * 1024.0));
    printf("Arithmetic intensity: ~0.10 FLOPs/byte (memory-bound)\n");
    
    double *A = (double *)malloc(n * n * sizeof(double));
    double *B = (double *)malloc(n * n * sizeof(double));
    
    init_array(n, A, B);
    
    double res_before = compute_residual(n, A);
    printf("Initial residual: %.6e\n", res_before);
    
    /* Run Jacobi iteration */
    jacobi_kernel(tsteps, n, A, B);
    
    double res_after = compute_residual(n, A);
    double checksum = compute_checksum(n, A);
    printf("Final residual: %.6e\n", res_after);
    printf("Checksum: %.6f\n", checksum);
    
    free(A);
    free(B);
    
    return 0;
}
