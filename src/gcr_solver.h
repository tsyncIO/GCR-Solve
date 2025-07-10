#ifndef GCR_SOLVER_H
#define GCR_SOLVER_H

#include "poisson.h" // Includes the Vector typedef

/**
 * @brief Copies the elements from one vector to another.
 * @param dest The destination vector.
 * @param src The source vector.
 * @param n The number of elements in the vectors.
 */
void vec_copy(Vector dest, const Vector src, int n);

/**
 * @brief Adds two vectors element-wise: res = v1 + v2.
 * @param res The result vector.
 * @param v1 The first operand vector.
 * @param v2 The second operand vector.
 * @param n The number of elements in the vectors.
 */
void vec_add(Vector res, const Vector v1, const Vector v2, int n);

/**
 * @brief Subtracts one vector from another element-wise: res = v1 - v2.
 * @param res The result vector.
 * @param v1 The first operand vector.
 * @param v2 The second operand vector.
 * @param n The number of elements in the vectors.
 */
void vec_subtract(Vector res, const Vector v1, const Vector v2, int n);

/**
 * @brief Scales a vector by a scalar: res = scalar * v.
 * @param res The result vector.
 * @param v The input vector.
 * @param scalar The scalar value to multiply by.
 * @param n The number of elements in the vectors.
 */
void vec_scale(Vector res, const Vector v, double scalar, int n);

/**
 * @brief Computes the dot product of two vectors: (v1, v2).
 * @param v1 The first vector.
 * @param v2 The second vector.
 * @param n The number of elements in the vectors.
 * @return The dot product value.
 */
double vec_dot(const Vector v1, const Vector v2, int n);

/**
 * @brief Computes the L2 norm (Euclidean norm) of a vector: ||v|| = sqrt((v,v)).
 * @param v The input vector.
 * @param n The number of elements in the vector.
 * @return The L2 norm value.
 */
double vec_norm(const Vector v, int n);

/**
 * @brief Solves a linear system A*y = b using the Generalized Conjugate Residual (GCR) method.
 * The A matrix is applied implicitly via the `apply_A_func` callback.
 * @param y On input, the initial guess for the solution; on output, the computed solution.
 * @param b The right-hand side vector.
 * @param internal_N The total number of internal grid points (size of y and b vectors).
 * @param apply_A_func A function pointer to the routine that applies the A matrix (e.g., apply_A from poisson.c).
 * @param rho_full_grid The full grid charge density (passed through to apply_A_func).
 * @param phi_boundary_full_grid The full grid boundary potential values (passed through to apply_A_func).
 * @param Nx Total grid points in x-direction (including boundaries, passed through to apply_A_func).
 * @param Ny Total grid points in y-direction (including boundaries, passed through to apply_A_func).
 * @param h Grid spacing (passed through to apply_A_func).
 * @param epsilon_0 Permittivity of free space (passed through to apply_A_func).
 * @param max_iterations Maximum number of GCR iterations to perform.
 * @param tolerance The relative residual norm tolerance for convergence (||r_k||/||b|| < tolerance).
 * @return 0 if the solver converged successfully, 1 otherwise.
 */
int solve_gcr(Vector y, const Vector b, int internal_N,
              void (*apply_A_func)(const Vector, Vector, const Vector, const Vector, int, int, double, double),
              const Vector rho_full_grid, const Vector phi_boundary_full_grid,
              int Nx, int Ny, double h, double epsilon_0,
              int max_iterations, double tolerance);

#endif // GCR_SOLVER_H
