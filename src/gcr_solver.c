#include "gcr_solver.h"
#include <stdio.h>
#include <stdlib.h> // For malloc, free, exit
#include <math.h>   // For sqrt
#include <omp.h>    // Required for OpenMP pragmas

/**
 * @brief Copies the elements from one vector to another.
 * Parallelized using OpenMP.
 */
void vec_copy(Vector dest, const Vector src, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        dest[i] = src[i];
    }
}

/**
 * @brief Adds two vectors element-wise: res = v1 + v2.
 * Parallelized using OpenMP.
 */
void vec_add(Vector res, const Vector v1, const Vector v2, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        res[i] = v1[i] + v2[i];
    }
}

/**
 * @brief Subtracts one vector from another element-wise: res = v1 - v2.
 * Parallelized using OpenMP.
 */
void vec_subtract(Vector res, const Vector v1, const Vector v2, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        res[i] = v1[i] - v2[i];
    }
}

/**
 * @brief Scales a vector by a scalar: res = scalar * v.
 * Parallelized using OpenMP.
 */
void vec_scale(Vector res, const Vector v, double scalar, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        res[i] = v[i] * scalar;
    }
}

/**
 * @brief Computes the dot product of two vectors: (v1, v2).
 * Parallelized using OpenMP with a reduction.
 */
double vec_dot(const Vector v1, const Vector v2, int n) {
    double sum = 0.0;
    // 'reduction(+:sum)' safely accumulates sum across threads.
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; ++i) {
        sum += v1[i] * v2[i];
    }
    return sum;
}

/**
 * @brief Computes the L2 norm (Euclidean norm) of a vector: ||v|| = sqrt((v,v)).
 * Leverages the parallelized vec_dot function.
 */
double vec_norm(const Vector v, int n) {
    return sqrt(vec_dot(v, v, n));
}

/**
 * @brief Solves a linear system A*y = b using the Generalized Conjugate Residual (GCR) method.
 * This implementation stores all previous p and Ap vectors (full GCR).
 * For very large problems/many iterations, memory usage can become an issue.
 */
int solve_gcr(Vector y, const Vector b, int internal_N,
              void (*apply_A_func)(const Vector, Vector, const Vector, const Vector, int, int, double, double),
              const Vector rho_full_grid, const Vector phi_boundary_full_grid,
              int Nx, int Ny, double h, double epsilon_0,
              int max_iterations, double tolerance) {

    // Allocate memory for temporary vectors used within the GCR iteration.
    Vector r = (Vector) malloc(internal_N * sizeof(double)); // Residual vector
    Vector Ap = (Vector) malloc(internal_N * sizeof(double)); // Stores A * p_k

    // GCR requires storing previous search directions (p_k) and their A-multiplied counterparts (Ap_k, often called q_k).
    // These are arrays of pointers to vectors.
    Vector *p_vectors = (Vector*) malloc(max_iterations * sizeof(Vector));
    Vector *Ap_vectors = (Vector*) malloc(max_iterations * sizeof(Vector));

    // Check for successful memory allocation.
    if (!r || !Ap || !p_vectors || !Ap_vectors) {
        fprintf(stderr, "Error: Memory allocation failed in solve_gcr for temporary vectors or history.\n");
        exit(EXIT_FAILURE);
    }

    // Calculate the norm of the right-hand side vector 'b'.
    // This is used for the relative residual convergence check.
    double b_norm = vec_norm(b, internal_N);
    if (b_norm < 1e-18) { // Handle case where b is effectively zero (trivial solution)
        fprintf(stderr, "Warning: Norm of b is very small. Solution might be trivial.\n");
        // If b is zero, the solution y should also be zero.
        vec_scale(y, y, 0.0, internal_N);
        // Free allocated memory before returning.
        free(r); free(Ap); free(p_vectors); free(Ap_vectors);
        return 0; // Success (trivial solution)
    }

    // 1. Compute initial residual r_0 = b - A*y_0
    // Call the provided 'apply_A_func' to compute A*y_0.
    apply_A_func(y, Ap, rho_full_grid, phi_boundary_full_grid, Nx, Ny, h, epsilon_0);
    // Subtract A*y_0 from b to get the initial residual.
    vec_subtract(r, b, Ap, internal_N);

    // Calculate and print the initial relative residual norm.
    double initial_residual_norm = vec_norm(r, internal_N);
    double relative_residual_norm = initial_residual_norm / b_norm;
    printf("Initial relative residual: %.2e\n", relative_residual_norm);

    int iter = 0; // Iteration counter
    // Main GCR iteration loop. Continues until convergence or max iterations reached.
    while (relative_residual_norm > tolerance && iter < max_iterations) {
        // Allocate memory for the current iteration's p_k and Ap_k (q_k) vectors.
        p_vectors[iter] = (Vector) malloc(internal_N * sizeof(double));
        Ap_vectors[iter] = (Vector) malloc(internal_N * sizeof(double));
        if (!p_vectors[iter] || !Ap_vectors[iter]) {
            fprintf(stderr, "Error: Memory allocation for p_k or Ap_k failed at iteration %d.\n", iter);
            // Free all memory allocated so far before exiting.
            free(r); free(Ap);
            for (int i = 0; i < iter; ++i) { free(p_vectors[i]); free(Ap_vectors[i]); }
            free(p_vectors); free(Ap_vectors);
            exit(EXIT_FAILURE);
        }

        // Set the initial search direction for this iteration: p_k = r_k
        vec_copy(p_vectors[iter], r, internal_N);

        // Orthogonalize p_k against all previous Ap_j vectors (Gram-Schmidt process).
        // This ensures the search directions are A-orthogonal.
        for (int j = 0; j < iter; ++j) {
            // Compute coefficient beta_k,j = (Ap_j, r_k) / (Ap_j, Ap_j)
            double beta_kj = vec_dot(Ap_vectors[j], r, internal_N) / vec_dot(Ap_vectors[j], Ap_vectors[j], internal_N);
            // Update p_k: p_k = p_k - beta_k,j * p_j
            vec_scale(Ap, p_vectors[j], beta_kj, internal_N); // Reuse Ap as a temporary for beta_kj * p_j
            vec_subtract(p_vectors[iter], p_vectors[iter], Ap, internal_N);
        }

        // Compute q_k = A * p_k (using the orthogonalized p_k).
        apply_A_func(p_vectors[iter], Ap_vectors[iter], rho_full_grid, phi_boundary_full_grid, Nx, Ny, h, epsilon_0);

        // Compute step size alpha_k = (r_k, q_k) / (q_k, q_k).
        double alpha_k = vec_dot(r, Ap_vectors[iter], internal_N) / vec_dot(Ap_vectors[iter], Ap_vectors[iter], internal_N);

        // Update solution: y_{k+1} = y_k + alpha_k * p_k
        vec_scale(Ap, p_vectors[iter], alpha_k, internal_N); // Reuse Ap as a temporary for alpha_k * p_k
        vec_add(y, y, Ap, internal_N);

        // Update residual: r_{k+1} = r_k - alpha_k * q_k
        vec_scale(Ap, Ap_vectors[iter], alpha_k, internal_N); // Reuse Ap as a temporary for alpha_k * q_k
        vec_subtract(r, r, Ap, internal_N);

        // Recalculate the relative residual norm for convergence check.
        relative_residual_norm = vec_norm(r, internal_N) / b_norm;
        printf("Iteration %d, relative residual: %.2e\n", iter + 1, relative_residual_norm);

        iter++; // Increment iteration counter
    }

    // --- Cleanup: Free all dynamically allocated memory ---
    free(r);
    free(Ap);
    // Free all stored p_vectors and Ap_vectors from the history.
    for (int i = 0; i < iter; ++i) {
        free(p_vectors[i]);
        free(Ap_vectors[i]);
    }
    free(p_vectors);
    free(Ap_vectors);

    // Return status based on convergence.
    if (relative_residual_norm <= tolerance) {
        printf("GCR converged successfully in %d iterations.\n", iter);
        return 0; // Success
    } else {
        printf("GCR did NOT converge within %d iterations. Final relative residual: %.2e\n", max_iterations, relative_residual_norm);
        return 1; // Failure
    }
}
