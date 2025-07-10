#ifndef UTILS_H
#define UTILS_H

#include "poisson.h" // Includes the Vector typedef

/**
 * @brief Allocates a 1D vector (array) of doubles.
 * @param n The number of elements to allocate.
 * @return A pointer to the allocated memory, or NULL if allocation fails.
 */
Vector allocate_vector(int n);

/**
 * @brief Frees memory allocated for a vector.
 * @param v The vector to free.
 */
void free_vector(Vector v);

/**
 * @brief Initializes all elements of a vector with a constant value.
 * @param v The vector to initialize.
 * @param n The number of elements in the vector.
 * @param val The value to set each element to.
 */
void initialize_vector(Vector v, int n, double val);

/**
 * @brief Writes the computed potential (phi) solution to an ASCII file.
 * This reconstructs the full grid (including boundary points) for output.
 * @param solution_phi The solution vector for internal points.
 * @param phi_boundary_full_grid The full grid boundary potential values.
 * @param Nx Total grid points in x-direction (including boundaries).
 * @param Ny Total grid points in y-direction (including boundaries).
 * @param h Grid spacing (not directly used for writing, but part of problem context).
 * @param filename The name of the file to write to.
 */
void write_solution_to_file(const Vector solution_phi, const Vector phi_boundary_full_grid,
                            int Nx, int Ny, double h, const char* filename);

/**
 * @brief Calculates the electric field components (Ex, Ey) from the potential solution
 * using finite differences and writes them to separate ASCII files.
 * Uses central differences where possible, and one-sided at boundaries.
 * @param solution_phi The solution vector for internal points.
 * @param phi_boundary_full_grid The full grid boundary potential values.
 * @param Nx Total grid points in x-direction (including boundaries).
 * @param Ny Total grid points in y-direction (including boundaries).
 * @param h Grid spacing.
 * @param filename_Ex The name of the file for the Ex component.
 * @param filename_Ey The name of the file for the Ey component.
 */
void calculate_and_write_electric_field(const Vector solution_phi, const Vector phi_boundary_full_grid,
                                        int Nx, int Ny, double h, const char* filename_Ex, const char* filename_Ey);

/**
 * @brief Starts a timer.
 * @return The current wall-clock time in seconds.
 */
double tic();

/**
 * @brief Stops a timer and calculates the elapsed time.
 * @param start_time The time returned by a previous call to tic().
 * @return The elapsed time in seconds.
 */
double toc(double start_time);

#endif // UTILS_H
