#include "utils.h"
#include "poisson.h" // For get_index
#include <stdio.h>   // For file I/O (fopen, fclose, fprintf)
#include <stdlib.h>  // For malloc, free, exit
#include <math.h>    // For sqrt (not directly used here, but common for numerical utils)
#include <omp.h>     // For omp_get_wtime for timing

/**
 * @brief Allocates a 1D vector (array) of doubles.
 * Includes error checking for memory allocation failure.
 * @param n The number of elements to allocate.
 * @return A pointer to the allocated memory. Exits on failure.
 */
Vector allocate_vector(int n) {
    Vector v = (Vector) malloc(n * sizeof(double));
    if (v == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for vector of size %d.\n", n);
        exit(EXIT_FAILURE); // Terminate program if memory allocation fails
    }
    return v;
}

/**
 * @brief Frees memory allocated for a vector.
 * Includes a check to prevent freeing a NULL pointer.
 * @param v The vector to free.
 */
void free_vector(Vector v) {
    if (v != NULL) {
        free(v);
    }
}

/**
 * @brief Initializes all elements of a vector with a constant value.
 * Parallelized using OpenMP.
 * @param v The vector to initialize.
 * @param n The number of elements in the vector.
 * @param val The value to set each element to.
 */
void initialize_vector(Vector v, int n, double val) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        v[i] = val;
    }
}

/**
 * @brief Writes the computed potential (phi) solution to an ASCII file.
 * The output file will represent the full grid, including boundary points,
 * in a format suitable for plotting (e.g., in MATLAB or Python).
 * @param solution_phi The solution vector for internal points.
 * @param phi_boundary_full_grid The full grid boundary potential values.
 * @param Nx Total grid points in x-direction (including boundaries).
 * @param Ny Total grid points in y-direction (including boundaries).
 * @param h Grid spacing (not directly used for writing, but part of problem context).
 * @param filename The name of the file to write to.
 */
void write_solution_to_file(const Vector solution_phi, const Vector phi_boundary_full_grid,
                            int Nx, int Ny, double h, const char* filename) {
    // Attempt to open the file for writing.
    FILE* fp = fopen(filename, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error: Could not open file %s for writing solution.\n", filename);
        return; // Return without writing if file cannot be opened
    }

    // Calculate dimensions of the internal grid.
    int internal_Nx = Nx - 2;
    // int internal_Ny = Ny - 2; // Not directly used in loop, but conceptually present

    // Iterate over the full grid (including boundaries) to reconstruct the solution field.
    for (int i = 0; i < Ny; ++i) {
        for (int j = 0; j < Nx; ++j) {
            double phi_val;
            // Check if the current point (i,j) is a boundary point.
            if (i == 0 || i == Ny - 1 || j == 0 || j == Nx - 1) {
                // If it's a boundary, use the fixed boundary potential value.
                phi_val = phi_boundary_full_grid[get_index(i, j, Nx)];
            } else {
                // If it's an internal point, get its value from the solution_phi vector.
                // Adjust index from (i,j) on full grid to (i-1, j-1) on internal grid.
                phi_val = solution_phi[(i - 1) * internal_Nx + (j - 1)];
            }
            // Write the potential value to the file, followed by a space.
            fprintf(fp, "%f ", phi_val);
        }
        // After each row, write a newline character.
        fprintf(fp, "\n");
    }
    // Close the file.
    fclose(fp);
    printf("Solution written to %s\n", filename);
}


/**
 * @brief Calculates the electric field components (Ex, Ey) from the potential solution
 * using finite differences and writes them to separate ASCII files.
 * E = -grad(phi), so Ex = -d(phi)/dx and Ey = -d(phi)/dy.
 * Uses central differences for internal points and one-sided differences at boundaries.
 * @param solution_phi The solution vector for internal points.
 * @param phi_boundary_full_grid The full grid boundary potential values.
 * @param Nx Total grid points in x-direction (including boundaries).
 * @param Ny Total grid points in y-direction (including boundaries).
 * @param h Grid spacing.
 * @param filename_Ex The name of the file for the Ex component.
 * @param filename_Ey The name of the file for the Ey component.
 */
void calculate_and_write_electric_field(const Vector solution_phi, const Vector phi_boundary_full_grid,
                                        int Nx, int Ny, double h, const char* filename_Ex, const char* filename_Ey) {
    // Open files for Ex and Ey components.
    FILE* fp_Ex = fopen(filename_Ex, "w");
    FILE* fp_Ey = fopen(filename_Ey, "w");

    if (fp_Ex == NULL || fp_Ey == NULL) {
        fprintf(stderr, "Error: Could not open E-field files for writing.\n");
        return;
    }

    // Calculate dimensions of the internal grid.
    int internal_Nx = Nx - 2;
    int internal_Ny = Ny - 2;

    // Create a temporary full grid including boundaries from the solution and boundary values.
    // This simplifies derivative calculations by providing a complete potential field.
    Vector full_phi_grid = allocate_vector(Nx * Ny);
    // Populate the full_phi_grid.
    for (int i = 0; i < Ny; ++i) {
        for (int j = 0; j < Nx; ++j) {
            if (i == 0 || i == Ny - 1 || j == 0 || j == Nx - 1) {
                full_phi_grid[get_index(i, j, Nx)] = phi_boundary_full_grid[get_index(i, j, Nx)];
            } else {
                full_phi_grid[get_index(i, j, Nx)] = solution_phi[(i - 1) * internal_Nx + (j - 1)];
            }
        }
    }

    // Iterate over the full grid to calculate Ex and Ey at each point.
    for (int i = 0; i < Ny; ++i) {
        for (int j = 0; j < Nx; ++j) {
            double Ex, Ey;

            // Calculate Ex = -d(phi)/dx using finite differences.
            // Use central difference for internal points, one-sided for boundaries.
            if (j == 0) { // Left boundary: use forward difference
                Ex = -(full_phi_grid[get_index(i, j + 1, Nx)] - full_phi_grid[get_index(i, j, Nx)]) / h;
            } else if (j == Nx - 1) { // Right boundary: use backward difference
                Ex = -(full_phi_grid[get_index(i, j, Nx)] - full_phi_grid[get_index(i, j - 1, Nx)]) / h;
            } else { // Internal points: use central difference
                Ex = -(full_phi_grid[get_index(i, j + 1, Nx)] - full_phi_grid[get_index(i, j - 1, Nx)]) / (2.0 * h);
            }

            // Calculate Ey = -d(phi)/dy using finite differences.
            // Use central difference for internal points, one-sided for boundaries.
            if (i == 0) { // Bottom boundary: use forward difference
                Ey = -(full_phi_grid[get_index(i + 1, j, Nx)] - full_phi_grid[get_index(i, j, Nx)]) / h;
            } else if (i == Ny - 1) { // Top boundary: use backward difference
                Ey = -(full_phi_grid[get_index(i, j, Nx)] - full_phi_grid[get_index(i - 1, j, Nx)]) / h;
            } else { // Internal points: use central difference
                Ey = -(full_phi_grid[get_index(i + 1, j, Nx)] - full_phi_grid[get_index(i - 1, j, Nx)]) / (2.0 * h);
            }

            // Write components to their respective files.
            fprintf(fp_Ex, "%f ", Ex);
            fprintf(fp_Ey, "%f ", Ey);
        }
        // Newline after each row.
        fprintf(fp_Ex, "\n");
        fprintf(fp_Ey, "\n");
    }

    // Free the temporary full potential grid.
    free_vector(full_phi_grid);
    // Close the files.
    fclose(fp_Ex);
    fclose(fp_Ey);
    printf("Electric field components written to %s and %s\n", filename_Ex, filename_Ey);
}

/**
 * @brief Starts a timer using OpenMP's high-resolution wall-clock time.
 * @return The current wall-clock time in seconds.
 */
double tic() {
    return omp_get_wtime();
}

/**
 * @brief Stops a timer and calculates the elapsed time.
 * @param start_time The time returned by a previous call to tic().
 * @return The elapsed time in seconds.
 */
double toc(double start_time) {
    return omp_get_wtime() - start_time;
}
