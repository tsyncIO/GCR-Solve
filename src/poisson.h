#ifndef POISSON_H
#define POISSON_H

// Define a type alias for a vector (a pointer to a double array).
// This improves readability and makes code more abstract.
typedef double* Vector;

/**
 * @brief Maps 2D grid coordinates (i,j) to a 1D array index.
 * @param i Row index (0 to Ny-1).
 * @param j Column index (0 to Nx-1).
 * @param Nx Total number of columns in the grid (including boundaries).
 * @return The 1D index corresponding to (i,j) in a row-major flattened array.
 */
int get_index(int i, int j, int Nx);

/**
 * @brief Applies the A matrix implicitly to an input vector.
 * This function computes `out_vec = A * in_vec`, where A represents
 * the discretized negative Laplacian operator.
 * It uses the 5-point finite difference stencil.
 * @param in_vec The input vector containing potential (phi) values for internal points.
 * @param out_vec The output vector to store the result of A * in_vec.
 * @param rho The charge density on the full grid (including boundary points, though usually zero there).
 * @param phi_boundary The fixed potential values on the boundary (full grid representation).
 * @param Nx Total number of grid points in the x-direction (columns), including boundaries.
 * @param Ny Total number of grid points in the y-direction (rows), including boundaries.
 * @param h Grid spacing (delta x = delta y).
 * @param epsilon_0 Permittivity of free space.
 */
void apply_A(const Vector in_vec, Vector out_vec, const Vector rho,
             const Vector phi_boundary, int Nx, int Ny, double h, double epsilon_0);

/**
 * @brief Creates the right-hand side (RHS) vector 'b' for the linear system A*y = b.
 * The 'b' vector depends on the charge density and contributions from boundary conditions.
 * @param b_vec The output vector to store the computed RHS values for internal points.
 * @param rho The charge density on the full grid (including boundary points).
 * @param phi_boundary The fixed potential values on the boundary (full grid representation).
 * @param Nx Total number of grid points in the x-direction (columns), including boundaries.
 * @param Ny Total number of grid points in the y-direction (rows), including boundaries.
 * @param h Grid spacing.
 * @param epsilon_0 Permittivity of free space.
 */
void create_b(Vector b_vec, const Vector rho, const Vector phi_boundary,
              int Nx, int Ny, double h, double epsilon_0);

#endif // POISSON_H
