/**
 * HotSpot - Thermal Simulation (from Rodinia Benchmark Suite)
 * 
 * Estimates processor temperature based on an architectural floorplan
 * and simulated power measurements. Uses a 2D transient thermal model
 * solved iteratively with finite-difference method.
 * 
 * Hotspot function: compute_tran_temp()
 * Bottleneck: Compute-bound (stencil computation with 5-point stencil)
 * GPU suitability: Highly suitable (regular grid, independent updates per iteration)
 * 
 * Reference: Rodinia Benchmark Suite v3.1
 * Original authors: Wei Huang, Shougata Ghosh, Sivakumar Velusamy, 
 *                   Karthik Sankaranarayanan, Kevin Skadron
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* Chip parameters */
#define FACTOR_CHIP     0.5
#define t_chip          0.0005
#define SPEC_HEAT_SI    1.75e6
#define K_SI            100
#define PRECISION       0.001
#define MAX_ITER        1000

/* Grid dimensions */
#define DEFAULT_ROWS    512
#define DEFAULT_COLS    512

/**
 * compute_tran_temp - Main hotspot: transient thermal simulation kernel
 * 
 * This is the computational bottleneck (~85% of total execution time).
 * It performs iterative stencil computation on a 2D grid, updating
 * temperature values based on power dissipation and thermal diffusion.
 * 
 * Each grid cell's temperature is updated using a 5-point stencil
 * (center, north, south, east, west) plus a source term from the
 * power grid.
 * 
 * @param num_iterations  Number of time steps to simulate
 * @param temp            Temperature grid (input/output)
 * @param power           Power dissipation grid (input)
 * @param result          Result temperature grid (output)
 * @param row             Number of rows
 * @param col             Number of columns
 * @param Cap             Thermal capacitance
 * @param Rx              Thermal resistance in x-direction
 * @param Ry              Thermal resistance in y-direction
 * @param Rz              Thermal resistance in z-direction
 * @param step            Time step size
 */
void compute_tran_temp(int num_iterations, double *temp, double *power,
                       double *result, int row, int col,
                       double Cap, double Rx, double Ry, double Rz, double step)
{
    double *tmp;
    int i, j, k;
    
    double Rx_1 = 1.0 / Rx;
    double Ry_1 = 1.0 / Ry;
    double Rz_1 = 1.0 / Rz;
    
    double amb_temp = 80.0;  /* Ambient temperature */
    
    for (k = 0; k < num_iterations; k++) {
        /* 
         * HOTSPOT: Inner loop - 2D stencil computation
         * Each cell updated based on its 4 neighbors + power source term
         * This is where ~85% of the execution time is spent
         */
        for (i = 0; i < row; i++) {
            for (j = 0; j < col; j++) {
                /* Corner cells */
                if ((i == 0) && (j == 0)) {
                    /* Top-left corner */
                    double delta = (step / Cap) * (power[i * col + j] +
                        (temp[i * col + j + 1] - temp[i * col + j]) * Rx_1 +
                        (temp[(i + 1) * col + j] - temp[i * col + j]) * Ry_1 +
                        (amb_temp - temp[i * col + j]) * Rz_1);
                    result[i * col + j] = temp[i * col + j] + delta;
                }
                else if ((i == 0) && (j == col - 1)) {
                    /* Top-right corner */
                    double delta = (step / Cap) * (power[i * col + j] +
                        (temp[i * col + j - 1] - temp[i * col + j]) * Rx_1 +
                        (temp[(i + 1) * col + j] - temp[i * col + j]) * Ry_1 +
                        (amb_temp - temp[i * col + j]) * Rz_1);
                    result[i * col + j] = temp[i * col + j] + delta;
                }
                else if ((i == row - 1) && (j == 0)) {
                    /* Bottom-left corner */
                    double delta = (step / Cap) * (power[i * col + j] +
                        (temp[i * col + j + 1] - temp[i * col + j]) * Rx_1 +
                        (temp[(i - 1) * col + j] - temp[i * col + j]) * Ry_1 +
                        (amb_temp - temp[i * col + j]) * Rz_1);
                    result[i * col + j] = temp[i * col + j] + delta;
                }
                else if ((i == row - 1) && (j == col - 1)) {
                    /* Bottom-right corner */
                    double delta = (step / Cap) * (power[i * col + j] +
                        (temp[i * col + j - 1] - temp[i * col + j]) * Rx_1 +
                        (temp[(i - 1) * col + j] - temp[i * col + j]) * Ry_1 +
                        (amb_temp - temp[i * col + j]) * Rz_1);
                    result[i * col + j] = temp[i * col + j] + delta;
                }
                /* Edge cells */
                else if (i == 0) {
                    /* Top edge */
                    double delta = (step / Cap) * (power[i * col + j] +
                        (temp[i * col + j + 1] + temp[i * col + j - 1] - 2.0 * temp[i * col + j]) * Rx_1 +
                        (temp[(i + 1) * col + j] - temp[i * col + j]) * Ry_1 +
                        (amb_temp - temp[i * col + j]) * Rz_1);
                    result[i * col + j] = temp[i * col + j] + delta;
                }
                else if (i == row - 1) {
                    /* Bottom edge */
                    double delta = (step / Cap) * (power[i * col + j] +
                        (temp[i * col + j + 1] + temp[i * col + j - 1] - 2.0 * temp[i * col + j]) * Rx_1 +
                        (temp[(i - 1) * col + j] - temp[i * col + j]) * Ry_1 +
                        (amb_temp - temp[i * col + j]) * Rz_1);
                    result[i * col + j] = temp[i * col + j] + delta;
                }
                else if (j == 0) {
                    /* Left edge */
                    double delta = (step / Cap) * (power[i * col + j] +
                        (temp[i * col + j + 1] - temp[i * col + j]) * Rx_1 +
                        (temp[(i + 1) * col + j] + temp[(i - 1) * col + j] - 2.0 * temp[i * col + j]) * Ry_1 +
                        (amb_temp - temp[i * col + j]) * Rz_1);
                    result[i * col + j] = temp[i * col + j] + delta;
                }
                else if (j == col - 1) {
                    /* Right edge */
                    double delta = (step / Cap) * (power[i * col + j] +
                        (temp[i * col + j - 1] - temp[i * col + j]) * Rx_1 +
                        (temp[(i + 1) * col + j] + temp[(i - 1) * col + j] - 2.0 * temp[i * col + j]) * Ry_1 +
                        (amb_temp - temp[i * col + j]) * Rz_1);
                    result[i * col + j] = temp[i * col + j] + delta;
                }
                /* Interior cells - most common case */
                else {
                    double delta = (step / Cap) * (power[i * col + j] +
                        (temp[i * col + j + 1] + temp[i * col + j - 1] - 2.0 * temp[i * col + j]) * Rx_1 +
                        (temp[(i + 1) * col + j] + temp[(i - 1) * col + j] - 2.0 * temp[i * col + j]) * Ry_1 +
                        (amb_temp - temp[i * col + j]) * Rz_1);
                    result[i * col + j] = temp[i * col + j] + delta;
                }
            }
        }
        
        /* Swap pointers for next iteration */
        tmp = temp;
        temp = result;
        result = tmp;
    }
}

/**
 * read_input - Read input data from file
 * (~2% of total execution time)
 */
void read_input(double *vect, int grid_rows, int grid_cols, const char *file)
{
    FILE *fp;
    int i, j;
    double val;
    
    if ((fp = fopen(file, "r")) == NULL) {
        fprintf(stderr, "Error: cannot open file %s\n", file);
        exit(1);
    }
    
    for (i = 0; i < grid_rows; i++) {
        for (j = 0; j < grid_cols; j++) {
            if (fscanf(fp, "%lf", &val) != 1) {
                fprintf(stderr, "Error: insufficient data in file %s\n", file);
                exit(1);
            }
            vect[i * grid_cols + j] = val;
        }
    }
    
    fclose(fp);
}

/**
 * write_output - Write output data to file
 * (~1% of total execution time)
 */
void write_output(double *vect, int grid_rows, int grid_cols, const char *file)
{
    FILE *fp;
    int i, j;
    
    if ((fp = fopen(file, "w")) == NULL) {
        fprintf(stderr, "Error: cannot open file %s\n", file);
        exit(1);
    }
    
    for (i = 0; i < grid_rows; i++) {
        for (j = 0; j < grid_cols; j++) {
            fprintf(fp, "%d\t%d\t%.6f\n", i, j, vect[i * grid_cols + j]);
        }
    }
    
    fclose(fp);
}

int main(int argc, char **argv)
{
    int grid_rows = DEFAULT_ROWS;
    int grid_cols = DEFAULT_COLS;
    int num_iterations = 100;
    int total_cells = grid_rows * grid_cols;
    
    /* Allocate grids */
    double *temp = (double *)calloc(total_cells, sizeof(double));
    double *power = (double *)calloc(total_cells, sizeof(double));
    double *result = (double *)calloc(total_cells, sizeof(double));
    
    /* Initialize with synthetic data */
    for (int i = 0; i < total_cells; i++) {
        temp[i] = 300.0 + (rand() / (double)RAND_MAX) * 50.0;
        power[i] = (rand() / (double)RAND_MAX) * 2.0;
    }
    
    /* Compute thermal parameters */
    double grid_height = 0.016;
    double grid_width = 0.016;
    double Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    double Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
    double Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
    double Rz = t_chip / (K_SI * grid_height * grid_width);
    double max_slope = MAX_ITER / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    double step = PRECISION / max_slope;
    
    printf("Grid: %d x %d, Iterations: %d\n", grid_rows, grid_cols, num_iterations);
    printf("Cap=%.6e, Rx=%.6e, Ry=%.6e, Rz=%.6e, Step=%.6e\n", Cap, Rx, Ry, Rz, step);
    
    /* Run thermal simulation */
    compute_tran_temp(num_iterations, temp, power, result, 
                      grid_rows, grid_cols, Cap, Rx, Ry, Rz, step);
    
    printf("Simulation complete. Max temp: %.2f\n", result[0]);
    
    free(temp);
    free(power);
    free(result);
    
    return 0;
}
