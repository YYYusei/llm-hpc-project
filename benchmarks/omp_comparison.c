/**
 * OpenMP Baseline Comparison
 * 
 * Applies simple OpenMP parallelisation to the same hotspot functions
 * that were translated to CUDA by the LLM. This provides a baseline
 * for evaluating the LLM-generated CUDA kernels' added value.
 *
 * Compile: gcc -O2 -fopenmp -o omp_comparison omp_comparison.c -lm
 * Run:     OMP_NUM_THREADS=8 ./omp_comparison
 *
 * Tests:
 *   1. miniMD Force (LJ potential) - embarrassingly parallel
 *   2. SPMV (sparse matrix-vector multiply) - row-parallel
 *   3. SYMGS (symmetric Gauss-Seidel) - data dependency challenge
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

/* ============================================================
 * 1. miniMD Lennard-Jones Force Computation
 * ============================================================ */

typedef struct {
    double x, y, z;
} Vec3;

/**
 * force_lj_serial - Serial baseline
 */
double force_lj_serial(const Vec3 *pos, Vec3 *force, const int *neighbors,
                       const int *num_neighbors, int natoms, double cutoff_sq,
                       double epsilon, double sigma)
{
    double sigma6 = sigma * sigma * sigma * sigma * sigma * sigma;
    double energy = 0.0;
    
    for (int i = 0; i < natoms; i++) {
        force[i].x = force[i].y = force[i].z = 0.0;
    }
    
    for (int i = 0; i < natoms; i++) {
        double fx = 0.0, fy = 0.0, fz = 0.0;
        
        for (int jj = 0; jj < num_neighbors[i]; jj++) {
            int j = neighbors[i * 100 + jj];  /* max 100 neighbors */
            
            double dx = pos[i].x - pos[j].x;
            double dy = pos[i].y - pos[j].y;
            double dz = pos[i].z - pos[j].z;
            double rsq = dx * dx + dy * dy + dz * dz;
            
            if (rsq < cutoff_sq && rsq > 1e-10) {
                double sr2 = 1.0 / rsq;
                double sr6 = sr2 * sr2 * sr2 * sigma6;
                double force_mag = 48.0 * epsilon * sr6 * (sr6 - 0.5) * sr2;
                
                fx += force_mag * dx;
                fy += force_mag * dy;
                fz += force_mag * dz;
                energy += 4.0 * epsilon * sr6 * (sr6 - 1.0);
            }
        }
        
        force[i].x = fx;
        force[i].y = fy;
        force[i].z = fz;
    }
    
    return energy * 0.5;
}

/**
 * force_lj_openmp - OpenMP parallelised version
 * Simple #pragma omp parallel for on outer atom loop
 */
double force_lj_openmp(const Vec3 *pos, Vec3 *force, const int *neighbors,
                       const int *num_neighbors, int natoms, double cutoff_sq,
                       double epsilon, double sigma)
{
    double sigma6 = sigma * sigma * sigma * sigma * sigma * sigma;
    double energy = 0.0;
    
    #pragma omp parallel for reduction(+:energy) schedule(dynamic, 64)
    for (int i = 0; i < natoms; i++) {
        double fx = 0.0, fy = 0.0, fz = 0.0;
        
        for (int jj = 0; jj < num_neighbors[i]; jj++) {
            int j = neighbors[i * 100 + jj];
            
            double dx = pos[i].x - pos[j].x;
            double dy = pos[i].y - pos[j].y;
            double dz = pos[i].z - pos[j].z;
            double rsq = dx * dx + dy * dy + dz * dz;
            
            if (rsq < cutoff_sq && rsq > 1e-10) {
                double sr2 = 1.0 / rsq;
                double sr6 = sr2 * sr2 * sr2 * sigma6;
                double force_mag = 48.0 * epsilon * sr6 * (sr6 - 0.5) * sr2;
                
                fx += force_mag * dx;
                fy += force_mag * dy;
                fz += force_mag * dz;
                energy += 4.0 * epsilon * sr6 * (sr6 - 1.0);
            }
        }
        
        force[i].x = fx;
        force[i].y = fy;
        force[i].z = fz;
    }
    
    return energy * 0.5;
}

/* ============================================================
 * 2. Sparse Matrix-Vector Multiply (SPMV)
 * ============================================================ */

/**
 * spmv_serial - Serial baseline
 */
void spmv_serial(int n, const int *rowptr, const int *colidx,
                 const double *values, const double *x, double *y)
{
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = rowptr[i]; j < rowptr[i + 1]; j++) {
            sum += values[j] * x[colidx[j]];
        }
        y[i] = sum;
    }
}

/**
 * spmv_openmp - OpenMP parallelised version
 * Row-parallel: each thread handles a subset of rows
 */
void spmv_openmp(int n, const int *rowptr, const int *colidx,
                 const double *values, const double *x, double *y)
{
    #pragma omp parallel for schedule(dynamic, 128)
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = rowptr[i]; j < rowptr[i + 1]; j++) {
            sum += values[j] * x[colidx[j]];
        }
        y[i] = sum;
    }
}

/* ============================================================
 * 3. Symmetric Gauss-Seidel (SYMGS)
 * ============================================================ */

/**
 * symgs_serial - Serial baseline (sequential by nature)
 */
void symgs_serial(int n, const int *rowptr, const int *colidx,
                  const double *values, const double *rhs, double *x)
{
    /* Forward sweep */
    for (int i = 0; i < n; i++) {
        double sum = rhs[i];
        double diag = 0.0;
        
        for (int j = rowptr[i]; j < rowptr[i + 1]; j++) {
            if (colidx[j] == i) {
                diag = values[j];
            } else {
                sum -= values[j] * x[colidx[j]];
            }
        }
        
        if (fabs(diag) > 1e-20) {
            x[i] = sum / diag;
        }
    }
    
    /* Backward sweep */
    for (int i = n - 1; i >= 0; i--) {
        double sum = rhs[i];
        double diag = 0.0;
        
        for (int j = rowptr[i]; j < rowptr[i + 1]; j++) {
            if (colidx[j] == i) {
                diag = values[j];
            } else {
                sum -= values[j] * x[colidx[j]];
            }
        }
        
        if (fabs(diag) > 1e-20) {
            x[i] = sum / diag;
        }
    }
}

/**
 * symgs_multicolor_openmp - OpenMP with multi-colouring
 * 
 * Multi-colouring breaks the sequential dependency:
 * nodes of the same color have no data dependencies between them
 * and can be processed in parallel.
 * 
 * This is the SAME strategy the LLM cascaded pipeline identified
 * for the CUDA version.
 */
void symgs_multicolor_openmp(int n, const int *rowptr, const int *colidx,
                             const double *values, const double *rhs, double *x,
                             int num_colors, const int *color_offsets,
                             const int *color_indices)
{
    /* Forward sweep: process colors in order */
    for (int c = 0; c < num_colors; c++) {
        int start = color_offsets[c];
        int end = color_offsets[c + 1];
        
        #pragma omp parallel for schedule(static)
        for (int ci = start; ci < end; ci++) {
            int i = color_indices[ci];
            double sum = rhs[i];
            double diag = 0.0;
            
            for (int j = rowptr[i]; j < rowptr[i + 1]; j++) {
                if (colidx[j] == i) {
                    diag = values[j];
                } else {
                    sum -= values[j] * x[colidx[j]];
                }
            }
            
            if (fabs(diag) > 1e-20) {
                x[i] = sum / diag;
            }
        }
    }
    
    /* Backward sweep: process colors in reverse order */
    for (int c = num_colors - 1; c >= 0; c--) {
        int start = color_offsets[c];
        int end = color_offsets[c + 1];
        
        #pragma omp parallel for schedule(static)
        for (int ci = start; ci < end; ci++) {
            int i = color_indices[ci];
            double sum = rhs[i];
            double diag = 0.0;
            
            for (int j = rowptr[i]; j < rowptr[i + 1]; j++) {
                if (colidx[j] == i) {
                    diag = values[j];
                } else {
                    sum -= values[j] * x[colidx[j]];
                }
            }
            
            if (fabs(diag) > 1e-20) {
                x[i] = sum / diag;
            }
        }
    }
}

/* ============================================================
 * Helper functions
 * ============================================================ */

void generate_lj_data(Vec3 **pos_out, int **neighbors_out, int **num_neighbors_out,
                      int natoms)
{
    Vec3 *pos = (Vec3 *)malloc(natoms * sizeof(Vec3));
    int *neighbors = (int *)calloc(natoms * 100, sizeof(int));
    int *num_neigh = (int *)calloc(natoms, sizeof(int));
    
    srand(42);
    double box = pow(natoms / 0.8, 1.0 / 3.0);
    
    for (int i = 0; i < natoms; i++) {
        pos[i].x = (double)rand() / RAND_MAX * box;
        pos[i].y = (double)rand() / RAND_MAX * box;
        pos[i].z = (double)rand() / RAND_MAX * box;
    }
    
    double cutoff = 2.5;
    double cutoff_sq = cutoff * cutoff;
    
    for (int i = 0; i < natoms; i++) {
        int count = 0;
        for (int j = 0; j < natoms && count < 100; j++) {
            if (i == j) continue;
            double dx = pos[i].x - pos[j].x;
            double dy = pos[i].y - pos[j].y;
            double dz = pos[i].z - pos[j].z;
            if (dx*dx + dy*dy + dz*dz < cutoff_sq) {
                neighbors[i * 100 + count] = j;
                count++;
            }
        }
        num_neigh[i] = count;
    }
    
    *pos_out = pos;
    *neighbors_out = neighbors;
    *num_neighbors_out = num_neigh;
}

void generate_sparse_matrix(int n, int bandwidth, int **rowptr_out, int **colidx_out,
                            double **values_out, int *nnz_out)
{
    int max_nnz = n * (2 * bandwidth + 1);
    int *rowptr = (int *)malloc((n + 1) * sizeof(int));
    int *colidx = (int *)malloc(max_nnz * sizeof(int));
    double *values = (double *)malloc(max_nnz * sizeof(double));
    
    int nnz = 0;
    for (int i = 0; i < n; i++) {
        rowptr[i] = nnz;
        int jmin = (i - bandwidth > 0) ? i - bandwidth : 0;
        int jmax = (i + bandwidth < n) ? i + bandwidth : n - 1;
        for (int j = jmin; j <= jmax; j++) {
            colidx[nnz] = j;
            values[nnz] = (i == j) ? 10.0 + (double)(i % 5) : -1.0 / (fabs(i-j) + 1.0);
            nnz++;
        }
    }
    rowptr[n] = nnz;
    
    *rowptr_out = rowptr;
    *colidx_out = colidx;
    *values_out = values;
    *nnz_out = nnz;
}

void simple_coloring(int n, int bandwidth, int *num_colors_out,
                     int **color_offsets_out, int **color_indices_out)
{
    /* Simple greedy coloring for banded matrix */
    int num_colors = 2 * bandwidth + 1;
    if (num_colors > n) num_colors = n;
    
    int *colors = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        colors[i] = i % num_colors;
    }
    
    int *offsets = (int *)calloc(num_colors + 1, sizeof(int));
    int *indices = (int *)malloc(n * sizeof(int));
    
    /* Count per color */
    for (int i = 0; i < n; i++) offsets[colors[i] + 1]++;
    for (int c = 0; c < num_colors; c++) offsets[c + 1] += offsets[c];
    
    /* Fill indices */
    int *pos = (int *)calloc(num_colors, sizeof(int));
    for (int c = 0; c < num_colors; c++) pos[c] = offsets[c];
    for (int i = 0; i < n; i++) {
        int c = colors[i];
        indices[pos[c]++] = i;
    }
    
    *num_colors_out = num_colors;
    *color_offsets_out = offsets;
    *color_indices_out = indices;
    
    free(colors);
    free(pos);
}

/* ============================================================
 * Benchmark driver
 * ============================================================ */

#define NRUNS 5

int main(int argc, char **argv)
{
    int num_threads = omp_get_max_threads();
    printf("OpenMP Comparison Benchmark\n");
    printf("Threads: %d\n\n", num_threads);
    
    /* -------- Test 1: miniMD Force -------- */
    {
        int natoms = 100000;
        Vec3 *pos, *force_s, *force_p;
        int *neighbors, *num_neighbors;
        
        generate_lj_data(&pos, &neighbors, &num_neighbors, natoms);
        force_s = (Vec3 *)calloc(natoms, sizeof(Vec3));
        force_p = (Vec3 *)calloc(natoms, sizeof(Vec3));
        
        double cutoff_sq = 2.5 * 2.5;
        
        /* Serial */
        double t0 = omp_get_wtime();
        for (int r = 0; r < NRUNS; r++)
            force_lj_serial(pos, force_s, neighbors, num_neighbors, natoms, cutoff_sq, 1.0, 1.0);
        double t_serial = (omp_get_wtime() - t0) / NRUNS;
        
        /* OpenMP */
        t0 = omp_get_wtime();
        for (int r = 0; r < NRUNS; r++)
            force_lj_openmp(pos, force_p, neighbors, num_neighbors, natoms, cutoff_sq, 1.0, 1.0);
        double t_omp = (omp_get_wtime() - t0) / NRUNS;
        
        /* Verify */
        double max_err = 0.0;
        for (int i = 0; i < natoms; i++) {
            max_err = fmax(max_err, fabs(force_s[i].x - force_p[i].x));
            max_err = fmax(max_err, fabs(force_s[i].y - force_p[i].y));
            max_err = fmax(max_err, fabs(force_s[i].z - force_p[i].z));
        }
        
        printf("1. miniMD Force (%d atoms)\n", natoms);
        printf("   Serial:  %.4f ms\n", t_serial * 1000);
        printf("   OpenMP:  %.4f ms  (%.2fx speedup)\n", t_omp * 1000, t_serial / t_omp);
        printf("   Max error: %.2e\n\n", max_err);
        
        free(pos); free(force_s); free(force_p);
        free(neighbors); free(num_neighbors);
    }
    
    /* -------- Test 2: SPMV -------- */
    {
        int n = 100000, bandwidth = 13;
        int *rowptr, *colidx, nnz;
        double *values;
        
        generate_sparse_matrix(n, bandwidth, &rowptr, &colidx, &values, &nnz);
        
        double *x = (double *)malloc(n * sizeof(double));
        double *y_s = (double *)calloc(n, sizeof(double));
        double *y_p = (double *)calloc(n, sizeof(double));
        for (int i = 0; i < n; i++) x[i] = 1.0;
        
        /* Serial */
        double t0 = omp_get_wtime();
        for (int r = 0; r < NRUNS * 10; r++)
            spmv_serial(n, rowptr, colidx, values, x, y_s);
        double t_serial = (omp_get_wtime() - t0) / (NRUNS * 10);
        
        /* OpenMP */
        t0 = omp_get_wtime();
        for (int r = 0; r < NRUNS * 10; r++)
            spmv_openmp(n, rowptr, colidx, values, x, y_p);
        double t_omp = (omp_get_wtime() - t0) / (NRUNS * 10);
        
        double max_err = 0.0;
        for (int i = 0; i < n; i++)
            max_err = fmax(max_err, fabs(y_s[i] - y_p[i]));
        
        printf("2. SPMV (%d rows, %d nnz)\n", n, nnz);
        printf("   Serial:  %.4f ms\n", t_serial * 1000);
        printf("   OpenMP:  %.4f ms  (%.2fx speedup)\n", t_omp * 1000, t_serial / t_omp);
        printf("   Max error: %.2e\n\n", max_err);
        
        free(x); free(y_s); free(y_p);
        free(rowptr); free(colidx); free(values);
    }
    
    /* -------- Test 3: SYMGS -------- */
    {
        int n = 50000, bandwidth = 13;
        int *rowptr, *colidx, nnz;
        double *values;
        
        generate_sparse_matrix(n, bandwidth, &rowptr, &colidx, &values, &nnz);
        
        double *rhs = (double *)malloc(n * sizeof(double));
        double *x_s = (double *)calloc(n, sizeof(double));
        double *x_mc = (double *)calloc(n, sizeof(double));
        for (int i = 0; i < n; i++) rhs[i] = 1.0;
        
        int num_colors;
        int *color_offsets, *color_indices;
        simple_coloring(n, bandwidth, &num_colors, &color_offsets, &color_indices);
        
        /* Serial */
        double t0 = omp_get_wtime();
        for (int r = 0; r < NRUNS; r++) {
            memset(x_s, 0, n * sizeof(double));
            symgs_serial(n, rowptr, colidx, values, rhs, x_s);
        }
        double t_serial = (omp_get_wtime() - t0) / NRUNS;
        
        /* OpenMP multi-color */
        t0 = omp_get_wtime();
        for (int r = 0; r < NRUNS; r++) {
            memset(x_mc, 0, n * sizeof(double));
            symgs_multicolor_openmp(n, rowptr, colidx, values, rhs, x_mc,
                                    num_colors, color_offsets, color_indices);
        }
        double t_omp = (omp_get_wtime() - t0) / NRUNS;
        
        /* Note: multi-color gives DIFFERENT results due to different update order */
        double max_diff = 0.0;
        for (int i = 0; i < n; i++)
            max_diff = fmax(max_diff, fabs(x_s[i] - x_mc[i]));
        
        printf("3. SYMGS (%d rows, %d colors)\n", n, num_colors);
        printf("   Serial:        %.4f ms\n", t_serial * 1000);
        printf("   OpenMP (MC):   %.4f ms  (%.2fx speedup)\n", t_omp * 1000, t_serial / t_omp);
        printf("   Max diff (expected non-zero due to ordering): %.2e\n\n", max_diff);
        
        free(rhs); free(x_s); free(x_mc);
        free(rowptr); free(colidx); free(values);
        free(color_offsets); free(color_indices);
    }
    
    printf("NOTE: Compare these OpenMP speedups against LLM-generated CUDA speedups\n");
    printf("to evaluate the added value of LLM GPU code generation.\n");
    
    return 0;
}
