/**
 * NAS CG - Conjugate Gradient Benchmark (from NAS Parallel Benchmarks)
 *
 * Solves an unstructured sparse linear system Ax = b using the
 * conjugate gradient method. The dominant operation is sparse
 * matrix-vector multiplication (SpMV) in CSR format.
 *
 * Hotspot function: sparse_matvec() - sparse matrix-vector multiply
 * Secondary hotspot: dot_product() - vector inner product (reduction)
 * Bottleneck: Memory-bound (irregular access through column indices, low arithmetic intensity)
 * GPU suitability: Partially suitable (SpMV benefits from GPU but irregular access limits efficiency)
 *
 * Reference: NAS Parallel Benchmarks 3.4
 * Class A: N=14000, NZ=1852000
 * Class B: N=75000, NZ=13000000
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* Problem size (Class A equivalent) */
#define NA          14000
#define NONZER      11
#define NZ_EST      (NA * (NONZER + 1) * (NONZER + 1))
#define NITER       15
#define SHIFT       10.0
#define ZETA_VERIFY 17.130235054029

/**
 * sparse_matvec - Sparse matrix-vector multiply (CSR format)
 *
 * PRIMARY HOTSPOT (~65% of total execution time).
 * Computes y = A * x where A is stored in CSR format.
 *
 * Memory-bound because:
 * - Indirect indexing through colidx[] causes irregular memory access
 * - Low arithmetic intensity: 2 FLOPs per memory access (1 mult + 1 add)
 * - Large matrix doesn't fit in cache
 * - Sequential access to values[] but random access to x[] via colidx[]
 *
 * @param n         Matrix dimension
 * @param rowptr    Row pointer array (n+1 elements, CSR format)
 * @param colidx    Column index array (nnz elements)
 * @param values    Non-zero value array (nnz elements)
 * @param x         Input vector (n elements)
 * @param y         Output vector (n elements)
 */
void sparse_matvec(int n, const int *rowptr, const int *colidx,
                   const double *values, const double *x, double *y)
{
    int i, j;
    
    for (i = 0; i < n; i++) {
        double sum = 0.0;
        
        /*
         * HOTSPOT: Inner loop - accumulate dot product for row i
         * Each iteration: 1 indirect load (x[colidx[j]]) + 1 multiply + 1 add
         * Arithmetic intensity: ~0.25 FLOPs/byte (very memory-bound)
         * The indirect access x[colidx[j]] causes cache misses
         */
        for (j = rowptr[i]; j < rowptr[i + 1]; j++) {
            sum += values[j] * x[colidx[j]];
        }
        
        y[i] = sum;
    }
}

/**
 * dot_product - Vector inner product with reduction
 *
 * SECONDARY HOTSPOT (~15% of total execution time).
 * Memory-bound: streaming access, 2 FLOPs per 16 bytes loaded.
 *
 * @param x, y      Input vectors (n elements each)
 * @param n         Vector length
 * @return          Sum of x[i] * y[i]
 */
double dot_product(const double *x, const double *y, int n)
{
    double sum = 0.0;
    
    for (int i = 0; i < n; i++) {
        sum += x[i] * y[i];
    }
    
    return sum;
}

/**
 * vec_axpy - Vector AXPY: y = alpha * x + y
 * (~8% of total execution time)
 */
void vec_axpy(double alpha, const double *x, double *y, int n)
{
    for (int i = 0; i < n; i++) {
        y[i] += alpha * x[i];
    }
}

/**
 * vec_copy - Vector copy: y = x
 * (~3% of total execution time) 
 */
void vec_copy(const double *x, double *y, int n)
{
    memcpy(y, x, n * sizeof(double));
}

/**
 * vec_scale - Vector scale: x = alpha * x
 * (~2% of total execution time)
 */
void vec_scale(double alpha, double *x, int n)
{
    for (int i = 0; i < n; i++) {
        x[i] *= alpha;
    }
}

/**
 * conj_grad - Conjugate gradient solver
 *
 * Solves the system Ax = b using CG iteration.
 * Calls sparse_matvec() and dot_product() as inner kernels.
 *
 * @param n         Problem dimension
 * @param rowptr    CSR row pointers
 * @param colidx    CSR column indices
 * @param values    CSR values
 * @param x         Solution vector (input/output)
 * @param b         Right-hand side vector
 * @param rnorm     Residual norm (output)
 * @param max_iter  Maximum iterations
 */
void conj_grad(int n, const int *rowptr, const int *colidx,
               const double *values, double *x, const double *b,
               double *rnorm, int max_iter)
{
    double *r = (double *)calloc(n, sizeof(double));
    double *p = (double *)calloc(n, sizeof(double));
    double *q = (double *)calloc(n, sizeof(double));
    
    /* r = b - A*x */
    sparse_matvec(n, rowptr, colidx, values, x, q);
    for (int i = 0; i < n; i++) {
        r[i] = b[i] - q[i];
        p[i] = r[i];
    }
    
    double rho = dot_product(r, r, n);
    
    for (int iter = 0; iter < max_iter; iter++) {
        /* q = A * p */
        sparse_matvec(n, rowptr, colidx, values, p, q);
        
        /* alpha = rho / (p' * q) */
        double pq = dot_product(p, q, n);
        double alpha = rho / (pq + 1e-20);
        
        /* x = x + alpha * p */
        vec_axpy(alpha, p, x, n);
        
        /* r = r - alpha * q */
        vec_axpy(-alpha, q, r, n);
        
        /* rho_new = r' * r */
        double rho_new = dot_product(r, r, n);
        
        /* Check convergence */
        if (sqrt(rho_new) < 1e-10) {
            printf("  Converged at iteration %d, residual = %.6e\n", iter, sqrt(rho_new));
            break;
        }
        
        /* beta = rho_new / rho */
        double beta = rho_new / (rho + 1e-20);
        
        /* p = r + beta * p */
        for (int i = 0; i < n; i++) {
            p[i] = r[i] + beta * p[i];
        }
        
        rho = rho_new;
    }
    
    *rnorm = sqrt(rho);
    
    free(r); free(p); free(q);
}

/**
 * generate_sparse_matrix - Create a synthetic sparse matrix in CSR format
 * Similar to NAS CG's matrix generation (banded + random structure)
 */
void generate_sparse_matrix(int n, int nonzer, int **rowptr_out, int **colidx_out,
                            double **values_out, int *nnz_out)
{
    /* Estimate max non-zeros */
    int max_nnz = n * (2 * nonzer + 1);
    int *rowptr = (int *)malloc((n + 1) * sizeof(int));
    int *colidx = (int *)malloc(max_nnz * sizeof(int));
    double *values = (double *)malloc(max_nnz * sizeof(double));
    
    int nnz = 0;
    srand(42);
    
    for (int i = 0; i < n; i++) {
        rowptr[i] = nnz;
        
        /* Band structure: include neighbors within NONZER distance */
        int jmin = (i - nonzer > 0) ? i - nonzer : 0;
        int jmax = (i + nonzer < n) ? i + nonzer : n - 1;
        
        for (int j = jmin; j <= jmax; j++) {
            colidx[nnz] = j;
            if (i == j) {
                values[nnz] = SHIFT + (double)(rand() % 100) / 100.0;
            } else {
                values[nnz] = -1.0 / (fabs(i - j) + 1.0);
            }
            nnz++;
        }
    }
    rowptr[n] = nnz;
    
    *rowptr_out = rowptr;
    *colidx_out = colidx;
    *values_out = values;
    *nnz_out = nnz;
}

int main(int argc, char **argv)
{
    int n = NA;
    int *rowptr, *colidx;
    double *values;
    int nnz;
    
    printf("NAS CG Benchmark (Class A equivalent)\n");
    printf("N = %d, NONZER = %d\n", n, NONZER);
    
    /* Generate sparse matrix */
    generate_sparse_matrix(n, NONZER, &rowptr, &colidx, &values, &nnz);
    printf("Matrix generated: %d non-zeros (%.1f per row)\n", nnz, (double)nnz / n);
    
    /* Allocate vectors */
    double *x = (double *)calloc(n, sizeof(double));
    double *b = (double *)malloc(n * sizeof(double));
    
    /* Initialize RHS */
    for (int i = 0; i < n; i++) {
        b[i] = 1.0;
        x[i] = 0.0;
    }
    
    /* Run CG iterations */
    double rnorm;
    for (int it = 0; it < NITER; it++) {
        printf("Outer iteration %d: ", it);
        conj_grad(n, rowptr, colidx, values, x, b, &rnorm, 25);
        
        /* Compute lambda = x'Ax / x'x */
        double *ax = (double *)calloc(n, sizeof(double));
        sparse_matvec(n, rowptr, colidx, values, x, ax);
        double xax = dot_product(x, ax, n);
        double xx = dot_product(x, x, n);
        double zeta = SHIFT + 1.0 / (xax / (xx + 1e-20) + 1e-20);
        printf("zeta = %.10f, rnorm = %.6e\n", zeta, rnorm);
        
        /* Normalize x */
        double norm = 1.0 / sqrt(xx + 1e-20);
        vec_scale(norm, x, n);
        free(ax);
    }
    
    free(x); free(b);
    free(rowptr); free(colidx); free(values);
    
    return 0;
}
