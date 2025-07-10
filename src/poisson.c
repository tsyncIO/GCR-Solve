#include "poisson.h"
#include <stdio.h>
#include <stdlib.h> // For malloc, free, exit
#include <math.h>   // For mathematical functions (not directly used in this file, but common)
#include <omp.h>    // Required for OpenMP pragmas

/**
 * @brief Maps 2D grid coordinates (i,j) to a 1D array index.
 * Assumes a row-major order for flattening the 2D grid into a 1D array.
 * @param i Row index.
 * @param j Column index.
 * @param Nx Total number of columns in the grid.
 * @return The 1D index.
 */
int get_index(int i, int j, int Nx) {
    return i * Nx + j;
}

/**
 * @brief Applies the A matrix implicitly (the negative Laplacian operator)
 * to an input vector representing the potential field.
 * The operation is `out_vec = A * in_vec`.
 * The A matrix is implicitly defined by the 5-point finite difference stencil:
 * A * phi_ij = (4*phi_ij - phi_i+1,j - phi_i-1,j - phi_i,j+1 - phi_i,j-1) / h^2
 * This function operates only on the internal grid points.
 * @param in_vec Input vector of potential values for internal points.
 * @param out_vec Output vector to store the result of the matrix-vector product.
 * @param rho Full grid charge density (used for boundary contributions in create_b, but passed here for consistency).
 * @param phi_boundary Full grid boundary potential values.
 * @param Nx Total grid points in x-direction (including boundaries).
 * @param Ny Total grid points in y-direction (including boundaries).
 * @param h Grid spacing.
 * @param epsilon_0 Permittivity of free space (not directly used here, but passed for consistency).
 */
void apply_A(const Vector in_vec, Vector out_vec, const Vector rho,
             const Vector phi_boundary, int Nx, int Ny, double h, double epsilon_0) {
    // Calculate the dimensions of the internal grid (excluding boundaries)
    int internal_Nx = Nx - 2;
    int internal_Ny = Ny - 2;

    // Allocate a temporary full grid to easily access neighbor values,
    // combining internal points from 'in_vec' and boundary points from 'phi_boundary'.
    // This simplifies the stencil application logic by providing a uniform 2D access.
    // For very large grids, consider optimizing this by calculating neighbor indices directly
    // from in_vec and phi_boundary without creating a full temporary grid.
    double *full_grid = (double*) malloc(Nx * Ny * sizeof(double));
    if (full_grid == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for full_grid in apply_A.\n");
        exit(EXIT_FAILURE);
    }

    // Populate the temporary full_grid:
    // Boundary values come from phi_boundary.
    // Internal values come from in_vec (which is a flattened 1D array of internal points).
    // OpenMP parallelization for populating the grid. 'collapse(2)' parallelizes both loops.
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < Ny; ++i) {
        for (int j = 0; j < Nx; ++j) {
            if (i == 0 || i == Ny - 1 || j == 0 || j == Nx - 1) {
                // This is a boundary point
                full_grid[get_index(i, j, Nx)] = phi_boundary[get_index(i, j, Nx)];
            } else {
                // This is an internal point. Map its (i,j) to the 1D 'in_vec' index.
                // The (i-1) and (j-1) adjust for the 0-indexed internal grid.
                full_grid[get_index(i, j, Nx)] = in_vec[(i - 1) * internal_Nx + (j - 1)];
            }
        }
    }

    // Apply the 5-point finite difference stencil to all internal points.
    // The result is stored in 'out_vec', which also represents only internal points.
    // OpenMP parallelization for the main computation loop. 'collapse(2)' parallelizes both loops.
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < Ny - 1; ++i) { // Iterate over internal rows (from 1 to Ny-2)
        for (int j = 1; j < Nx - 1; ++j) { // Iterate over internal columns (from 1 to Nx-2)
            // Calculate the 1D index for the current internal point in 'out_vec'.
            int current_idx_flat = (i - 1) * internal_Nx + (j - 1);

            // Get phi values for the current point and its four direct neighbors from the full_grid.
            double phi_center = full_grid[get_index(i, j, Nx)];
            double phi_up     = full_grid[get_index(i + 1, j, Nx)];
            double phi_down   = full_grid[get_index(i - 1, j, Nx)];
            double phi_left   = full_grid[get_index(i, j - 1, Nx)];
            double phi_right  = full_grid[get_index(i, j + 1, Nx)];

            // Compute the A*phi value based on the finite difference approximation of -Laplacian.
            // The equation is: -(-4*phi_ij + phi_ip1j + phi_im1j + phi_ijp1 + phi_ijm1) / h^2
            // Which simplifies to: (4*phi_ij - phi_ip1j - phi_im1j - phi_ijp1 - phi_ijm1) / h^2
            out_vec[current_idx_flat] = (4.0 * phi_center - phi_up - phi_down - phi_left - phi_right) / (h * h);
        }
    }

    // Free the temporary full grid memory.
    free(full_grid);
}

/**
 * @brief Creates the right-hand side (RHS) vector 'b' for the linear system A*y = b.
 * The Poisson equation is -Delta(phi) = rho / epsilon_0.
 * Discretized: -(-4*phi_ij + phi_ip1j + phi_im1j + phi_ijp1 + phi_ijm1) / h^2 = rho_ij / epsilon_0
 * Rearranging for A*y = b, where 'y' contains internal phi values:
 * (4*phi_ij - phi_ip1j - phi_im1j - phi_ijp1 - phi_ijm1) / h^2 = rho_ij / epsilon_0
 * If a neighbor is a boundary point, its known phi value is moved to the RHS.
 * So, b_ij = (rho_ij / epsilon_0) + (boundary_neighbor_phi / h^2) terms
 * @param b_vec The output vector to store the computed RHS values for internal points.
 * @param rho The charge density on the full grid.
 * @param phi_boundary The fixed potential values on the boundary (full grid representation).
 * @param Nx Total number of grid points in the x-direction (including boundaries).
 * @param Ny Total number of grid points in the y-direction (rows), including boundaries.
 * @param h Grid spacing.
 * @param epsilon_0 Permittivity of free space.
 */
void create_b(Vector b_vec, const Vector rho, const Vector phi_boundary,
              int Nx, int Ny, double h, double epsilon_0) {
    // Calculate the dimensions of the internal grid.
    int internal_Nx = Nx - 2;
    int internal_Ny = Ny - 2;

    // OpenMP parallelization for creating the RHS vector. 'collapse(2)' parallelizes both loops.
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < Ny - 1; ++i) { // Iterate over internal rows
        for (int j = 1; j < Nx - 1; ++j) { // Iterate over internal columns
            // Calculate the 1D index for the current internal point in 'b_vec'.
            int current_idx_flat = (i - 1) * internal_Nx + (j - 1);

            // Start with the primary RHS term from the Poisson equation: rho_ij / epsilon_0
            double rhs_val = rho[get_index(i, j, Nx)] / epsilon_0;

            // Add contributions from boundary neighbors.
            // If a neighbor is on the boundary, its fixed potential contributes to the RHS.
            // The terms are phi_boundary_neighbor / h^2.
            if (i + 1 == Ny - 1) { // Top neighbor is boundary
                rhs_val += phi_boundary[get_index(i + 1, j, Nx)] / (h * h);
            }
            if (i - 1 == 0) { // Bottom neighbor is boundary
                rhs_val += phi_boundary[get_index(i - 1, j, Nx)] / (h * h);
            }
            if (j + 1 == Nx - 1) { // Right neighbor is boundary
                rhs_val += phi_boundary[get_index(i, j + 1, Nx)] / (h * h);
            }
            if (j - 1 == 0) { // Left neighbor is boundary
                rhs_val += phi_boundary[get_index(i, j - 1, Nx)] / (h * h);
            }

            // Store the computed RHS value.
            b_vec[current_idx_flat] = rhs_val;
        }
    }
}
