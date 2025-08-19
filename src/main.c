#include <stdio.h>     // For standard input/output functions (printf, fprintf)
#include <stdlib.h>   // For atoi, malloc, free, exit
#include <math.h>     // For mathematical functions (e.g., sqrt)
#include <omp.h>      // For OpenMP functions like omp_set_num_threads, omp_get_wtime

// Define problem constants
#define DEFAULT_NX 102       // Default number of grid points in x-direction (including boundaries)
#define DEFAULT_NY 102       // Default number of grid points in y-direction (including boundaries)
#define H_SPACING 0.01       // Grid spacing (delta x = delta y) in meters
#define EPSILON_0 8.854e-12  // Permittivity of free space (F/m)
#define DEFAULT_VOLTAGE 10.0 // Default voltage for capacitor plates in Volts

// Define GCR solver parameters
#define MAX_GCR_ITERATIONS 12000 // Maximum iterations for the GCR solver
#define GCR_TOLERANCE 1e-12      // Relative residual norm tolerance for convergence

// User-defined type alias for a vector
typedef double* Vector;

// --- Memory Management and Utility Functions ---

/**
 * @brief Allocates memory for a vector of doubles.
 * @param n The number of elements in the vector.
 * @return A pointer to the allocated memory. Exits on failure.
 */
Vector allocate_vector(int n) {
    Vector v = (Vector)malloc(n * sizeof(double));
    if (v == NULL) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }
    return v;
}

/**
 * @brief Frees the memory allocated for a vector.
 * @param v The vector pointer to free.
 */
void free_vector(Vector v) {
    if (v != NULL) {
        free(v);
    }
}

/**
 * @brief Initializes a vector with a constant value.
 * @param v The vector to initialize.
 * @param n The number of elements.
 * @param val The value to set for all elements.
 */
void initialize_vector(Vector v, int n, double val) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        v[i] = val;
    }
}

/**
 * @brief Maps 2D grid coordinates (i, j) to a 1D index for a row-major array.
 * @param i The row index.
 * @param j The column index.
 * @param Nx The total number of columns in the grid.
 * @return The 1D index.
 */
int get_index(int i, int j, int Nx) {
    return i * Nx + j;
}

/**
 * @brief Starts a timer and returns the current time.
 * @return The current time in seconds.
 */
double tic() {
    return omp_get_wtime();
}

/**
 * @brief Stops a timer and returns the elapsed time since start_time.
 * @param start_time The starting time.
 * @return The elapsed time in seconds.
 */
double toc(double start_time) {
    return omp_get_wtime() - start_time;
}

// --- Vector Operations for GCR Solver ---

/**
 * @brief Copies the elements from one vector to another.
 * @param dest The destination vector.
 * @param src The source vector.
 * @param n The number of elements.
 */
void vec_copy(Vector dest, const Vector src, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        dest[i] = src[i];
    }
}

/**
 * @brief Adds two vectors element-wise: res = v1 + v2.
 * @param res The result vector.
 * @param v1 The first vector.
 * @param v2 The second vector.
 * @param n The number of elements.
 */
void vec_add(Vector res, const Vector v1, const Vector v2, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        res[i] = v1[i] + v2[i];
    }
}

/**
 * @brief Subtracts one vector from another element-wise: res = v1 - v2.
 * @param res The result vector.
 * @param v1 The first vector.
 * @param v2 The second vector.
 * @param n The number of elements.
 */
void vec_subtract(Vector res, const Vector v1, const Vector v2, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        res[i] = v1[i] - v2[i];
    }
}

/**
 * @brief Computes the dot product of two vectors.
 * @param v1 The first vector.
 * @param v2 The second vector.
 * @param n The number of elements.
 * @return The dot product.
 */
double vec_dot(const Vector v1, const Vector v2, int n) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; ++i) {
        sum += v1[i] * v2[i];
    }
    return sum;
}

/**
 * @brief Computes the L2 norm (magnitude) of a vector.
 * @param v The vector.
 * @param n The number of elements.
 * @return The L2 norm.
 */
double vec_norm(const Vector v, int n) {
    return sqrt(vec_dot(v, v, n));
}

// --- Problem Setup Functions ---

/**
 * @brief Sets up the charge density (rho) on the full grid.
 * For this example, it's set to zero everywhere since the capacitor plates
 * are handled by boundary conditions.
 * @param rho The vector representing the charge density on the full grid (Nx * Ny).
 * @param Nx Total grid points in x-direction.
 * @param Ny Total grid points in y-direction.
 */
void setup_charge_density(Vector rho, int Nx, int Ny) {
    initialize_vector(rho, Nx * Ny, 0.0);
}

/**
 * @brief Sets up the boundary conditions for the electric potential (phi) on the full grid.
 * This function sets up a parallel-plate capacitor.
 * @param phi_boundary The vector representing the potential on the full grid (Nx * Ny).
 * @param Nx Total grid points in x-direction.
 * @param Ny Total grid points in y-direction.
 * @param voltage The voltage difference between the plates.
 */
void setup_boundary_conditions(Vector phi_boundary, int Nx, int Ny, double voltage) {
    // Initialize all potential values on the full grid to zero.
    initialize_vector(phi_boundary, Nx * Ny, 0.0);
    double half_voltage = voltage / 2.0;
    
    // Set top boundary (i=Ny-1) to +half_voltage
    for (int j = 0; j < Nx; ++j) {
        phi_boundary[get_index(Ny - 1, j, Nx)] = half_voltage;
    }
    
    // Set bottom boundary (i=0) to -half_voltage
    for (int j = 0; j < Nx; ++j) {
        phi_boundary[get_index(0, j, Nx)] = -half_voltage;
    }

    // Left and right boundaries remain at 0.0 from initialization.
}

/**
 * @brief Creates the right-hand side vector 'b' for the linear system.
 * The system is Ax = b, where A is the discretized Laplacian and x is the internal potential.
 * b is derived from the charge density and boundary conditions.
 * @param b_rhs The vector to store the result.
 * @param rho The charge density vector.
 * @param phi_boundary The boundary potential vector.
 * @param Nx The number of columns.
 * @param Ny The number of rows.
 * @param h The grid spacing.
 * @param epsilon_0 The permittivity of free space.
 */
void create_b(Vector b_rhs, const Vector rho, const Vector phi_boundary, int Nx, int Ny, double h, double epsilon_0) {
    int internal_Nx = Nx - 2;
    int internal_Ny = Ny - 2;
    int internal_N = internal_Nx * internal_Ny;
    
    double h_sq = h * h;
    double inv_epsilon = 1.0 / epsilon_0;

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < Ny - 1; ++i) {
        for (int j = 1; j < Nx - 1; ++j) {
            int internal_idx = get_index(i - 1, j - 1, internal_Nx);
            int full_idx = get_index(i, j, Nx);
            b_rhs[internal_idx] = -rho[full_idx] * h_sq * inv_epsilon;
            
            // Adjust for boundary conditions
            if (i == 1) { // Top row of internal points
                 b_rhs[internal_idx] -= phi_boundary[get_index(0, j, Nx)];
            }
            if (i == Ny - 2) { // Bottom row of internal points
                 b_rhs[internal_idx] -= phi_boundary[get_index(Ny-1, j, Nx)];
            }
            if (j == 1) { // Left-most column of internal points
                 b_rhs[internal_idx] -= phi_boundary[get_index(i, 0, Nx)];
            }
            if (j == Nx - 2) { // Right-most column of internal points
                 b_rhs[internal_idx] -= phi_boundary[get_index(i, Nx-1, Nx)];
            }
        }
    }
}

// --- Core GCR Solver Functionality ---

/**
 * @brief Applies the discretized 2D Laplacian operator A to a vector phi.
 * This is the central operation of the GCR solver.
 * A*phi = (-4*phi[i,j] + phi[i+1,j] + phi[i-1,j] + phi[i,j+1] + phi[i,j-1])
 * @param in_vec The input vector (internal potential).
 * @param out_vec The resulting vector after applying A.
 * @param Nx The number of columns in the full grid.
 * @param Ny The number of rows in the full grid.
 */
void apply_A(const Vector in_vec, Vector out_vec, int Nx, int Ny) {
    int internal_Nx = Nx - 2;
    int internal_Ny = Ny - 2;
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < internal_Ny; ++i) {
        for (int j = 0; j < internal_Nx; ++j) {
            int internal_idx = get_index(i, j, internal_Nx);
            double center = in_vec[internal_idx];
            double up = (i > 0) ? in_vec[get_index(i - 1, j, internal_Nx)] : 0.0;
            double down = (i < internal_Ny - 1) ? in_vec[get_index(i + 1, j, internal_Nx)] : 0.0;
            double left = (j > 0) ? in_vec[get_index(i, j - 1, internal_Nx)] : 0.0;
            double right = (j < internal_Nx - 1) ? in_vec[get_index(i, j + 1, internal_Nx)] : 0.0;
            
            out_vec[internal_idx] = -4.0 * center + up + down + left + right;
        }
    }
}

/**
 * @brief Solves the linear system using the GCR method.
 * @param y The solution vector (internal potential).
 * @param b The right-hand side vector.
 * @param internal_N The size of the internal grid (Nx-2)*(Ny-2).
 * @param apply_A_func Function pointer to apply the A matrix.
 * @param rho_full_grid Unused in GCR, but part of the function signature.
 * @param phi_boundary_full_grid Unused in GCR, but part of the function signature.
 * @param Nx The number of columns in the full grid.
 * @param Ny The number of rows in the full grid.
 * @param h Unused in GCR, but part of the function signature.
 * @param epsilon_0 Unused in GCR, but part of the function signature.
 * @param max_iter The maximum number of iterations.
 * @param tol The convergence tolerance.
 * @return The number of iterations or -1 on failure.
 */
int solve_gcr(Vector y, const Vector b, int internal_N, 
              void (*apply_A_func)(const Vector, Vector, int, int),
              const Vector rho_full_grid, const Vector phi_boundary_full_grid,
              int Nx, int Ny, double h, double epsilon_0,
              int max_iter, double tol) {
    
    Vector r = allocate_vector(internal_N);
    Vector p = allocate_vector(internal_N);
    Vector Ap = allocate_vector(internal_N);

    // Initial residual: r = b - Ay
    apply_A_func(y, Ap, Nx, Ny);
    vec_subtract(r, b, Ap, internal_N);
    
    double initial_norm = vec_norm(r, internal_N);
    double current_norm = initial_norm;
    
    printf("Initial residual norm: %.6e\n", initial_norm);
    
    // Store history of search directions to enforce orthogonality
    Vector *p_history = (Vector*)malloc(max_iter * sizeof(Vector));
    Vector *Ap_history = (Vector*)malloc(max_iter * sizeof(Vector));
    if (p_history == NULL || Ap_history == NULL) {
        fprintf(stderr, "Error: Memory allocation for history failed.\n");
        exit(EXIT_FAILURE);
    }

    int iter = 0;
    while (iter < max_iter && current_norm / initial_norm > tol) {
        
        // p_i = r_i - sum_{j=0}^{i-1} ( (A*p_i, A*p_j) / (A*p_j, A*p_j) ) * p_j
        vec_copy(p, r, internal_N);
        
        for (int i = 0; i < iter; ++i) {
            double alpha = vec_dot(Ap_history[i], r, internal_N) / vec_dot(Ap_history[i], Ap_history[i], internal_N);
            #pragma omp parallel for
            for(int k = 0; k < internal_N; ++k) {
                p[k] -= alpha * p_history[i][k];
            }
        }

        // Apply matrix A to the new search direction p
        apply_A_func(p, Ap, Nx, Ny);
        
        // Store current vectors in history
        p_history[iter] = allocate_vector(internal_N);
        vec_copy(p_history[iter], p, internal_N);
        
        Ap_history[iter] = allocate_vector(internal_N);
        vec_copy(Ap_history[iter], Ap, internal_N);

        // Update solution and residual
        double alpha = vec_dot(r, Ap, internal_N) / vec_dot(Ap, Ap, internal_N);
        #pragma omp parallel for
        for(int k = 0; k < internal_N; ++k) {
            y[k] += alpha * p[k];
            r[k] -= alpha * Ap[k];
        }

        current_norm = vec_norm(r, internal_N);
        if ((iter + 1) % 100 == 0) {
            printf("Iteration %d, residual norm: %.6e\n", iter + 1, current_norm);
        }
        
        iter++;
    }

    // Free history memory
    for (int i = 0; i < iter; ++i) {
        free_vector(p_history[i]);
        free_vector(Ap_history[i]);
    }
    free(p_history);
    free(Ap_history);
    free_vector(r);
    free_vector(p);
    free_vector(Ap);

    if (current_norm / initial_norm <= tol) {
        return iter; // Success
    } else {
        return -1; // Failure
    }
}

// --- Output Functions ---

/**
 * @brief Writes the full electric potential grid to a file.
 * @param solution_phi The internal solution vector from the solver.
 * @param phi_boundary_full_grid The full grid with boundary values.
 * @param Nx The number of columns.
 * @param Ny The number of rows.
 * @param h The grid spacing.
 * @param filename The output file name.
 */
void write_solution_to_file(const Vector solution_phi, const Vector phi_boundary_full_grid, int Nx, int Ny, double h, const char* filename) {
    FILE* fp = fopen(filename, "w");
    if (fp == NULL) {
        perror("Error opening file");
        return;
    }

    // Write the header with dimensions and spacing
    fprintf(fp, "%d %d %.6f\n", Nx, Ny, h);

    // Reconstruct the full grid from the solution and boundary values
    for (int i = 0; i < Ny; ++i) {
        for (int j = 0; j < Nx; ++j) {
            double phi_val;
            if (i == 0 || i == Ny - 1 || j == 0 || j == Nx - 1) {
                phi_val = phi_boundary_full_grid[get_index(i, j, Nx)];
            } else {
                phi_val = solution_phi[get_index(i-1, j-1, Nx-2)];
            }
            fprintf(fp, "%.6f ", phi_val);
        }
        fprintf(fp, "\n");
    }

    printf("Writing solution to %s\n", filename);
    fclose(fp);
}

/**
 * @brief Calculates the electric field from the potential and writes it to files.
 * @param sol The internal solution vector for the potential.
 * @param bound The full boundary potential grid.
 * @param Nx The number of columns.
 * @param Ny The number of rows.
 * @param h The grid spacing.
 * @param filename_Ex The output file name for the x-component.
 * @param filename_Ey The output file name for the y-component.
 */
void calculate_and_write_electric_field(const Vector sol, const Vector bound, int Nx, int Ny, double h, const char* filename_Ex, const char* filename_Ey) {
    FILE* fp_Ex = fopen(filename_Ex, "w");
    FILE* fp_Ey = fopen(filename_Ey, "w");
    if (fp_Ex == NULL || fp_Ey == NULL) {
        perror("Error opening E-field files");
        if (fp_Ex) fclose(fp_Ex);
        if (fp_Ey) fclose(fp_Ey);
        return;
    }
    
    // Write headers for both files
    fprintf(fp_Ex, "%d %d %.6f\n", Nx, Ny, h);
    fprintf(fp_Ey, "%d %d %.6f\n", Nx, Ny, h);

    printf("Calculating and writing electric field to %s and %s\n", filename_Ex, filename_Ey);

    // First, reconstruct the full potential grid from the internal solution and boundaries
    Vector phi_full_grid = allocate_vector(Nx * Ny);
    for (int i = 0; i < Ny; ++i) {
        for (int j = 0; j < Nx; ++j) {
            if (i == 0 || i == Ny - 1 || j == 0 || j == Nx - 1) {
                phi_full_grid[get_index(i, j, Nx)] = bound[get_index(i, j, Nx)];
            } else {
                phi_full_grid[get_index(i, j, Nx)] = sol[get_index(i - 1, j - 1, Nx-2)];
            }
        }
    }

    // Calculate E-field using central difference
    for (int i = 0; i < Ny; ++i) {
        for (int j = 0; j < Nx; ++j) {
            double Ex, Ey;

            // Use forward/backward difference at boundaries to avoid out-of-bounds access
            if (j == 0) Ex = -(phi_full_grid[get_index(i, j + 1, Nx)] - phi_full_grid[get_index(i, j, Nx)]) / h;
            else if (j == Nx - 1) Ex = -(phi_full_grid[get_index(i, j, Nx)] - phi_full_grid[get_index(i, j - 1, Nx)]) / h;
            else Ex = -(phi_full_grid[get_index(i, j + 1, Nx)] - phi_full_grid[get_index(i, j - 1, Nx)]) / (2.0 * h);

            if (i == 0) Ey = -(phi_full_grid[get_index(i + 1, j, Nx)] - phi_full_grid[get_index(i, j, Nx)]) / h;
            else if (i == Ny - 1) Ey = -(phi_full_grid[get_index(i, j, Nx)] - phi_full_grid[get_index(i - 1, j, Nx)]) / h;
            else Ey = -(phi_full_grid[get_index(i + 1, j, Nx)] - phi_full_grid[get_index(i - 1, j, Nx)]) / (2.0 * h);
            
            fprintf(fp_Ex, "%.6f ", Ex);
            fprintf(fp_Ey, "%.6f ", Ey);
        }
        fprintf(fp_Ex, "\n");
        fprintf(fp_Ey, "\n");
    }

    free_vector(phi_full_grid);
    fclose(fp_Ex);
    fclose(fp_Ey);
}

/**
 * @brief Writes problem parameters (Nx, Ny, h) to a text file in the current working directory.
 * @param Nx Number of grid points in x-direction.
 * @param Ny Number of grid points in y-direction.
 * @param h Grid spacing.
 */
void write_params_to_file(int Nx, int Ny, double h) {
    const char* filename = "params.txt";
    FILE* fp = fopen(filename, "w");
    if (fp == NULL) {
        perror("Error opening parameter file");
        return;
    }
    
    fprintf(fp, "Nx: %d\n", Nx);
    fprintf(fp, "Ny: %d\n", Ny);
    fprintf(fp, "h: %.6f\n", h);
    
    fclose(fp);
    printf("Wrote simulation parameters to %s\n", filename);
}

// --- Main Program Execution ---

int main(int argc, char *argv[]) {
    // Initialize grid dimensions and physical constants with default values.
    int Nx = DEFAULT_NX;
    int Ny = DEFAULT_NY;
    double h = H_SPACING;
    double epsilon_0 = EPSILON_0;
    double voltage = DEFAULT_VOLTAGE;

    // Parse command-line arguments for grid dimensions (Nx, Ny) and threads.
    if (argc >= 3) {
        Nx = atoi(argv[1]);
        Ny = atoi(argv[2]);
        if (Nx < 3 || Ny < 3) {
            fprintf(stderr, "Error: Nx and Ny must be at least 3.\n");
            return 1;
        }
        printf("Using grid size: Nx=%d, Ny=%d\n", Nx, Ny);
    } else {
        printf("Using default grid size: Nx=%d, Ny=%d\n", Nx, Ny);
        printf("Usage: %s [Nx] [Ny] [num_threads] [voltage]\n", argv[0]);
    }
    
    if (argc >= 4) {
        int num_threads = atoi(argv[3]);
        if (num_threads > 0) {
            omp_set_num_threads(num_threads);
            printf("Set OpenMP threads to: %d\n", num_threads);
        } else {
            fprintf(stderr, "Error: Number of threads must be positive.\n");
            return 1;
        }
    } else {
        printf("Using default OpenMP threads.\n");
    }

    if (argc == 5) {
        voltage = atof(argv[4]);
        printf("Using capacitor voltage: %.2f V\n", voltage);
    } else {
        printf("Using default capacitor voltage: %.2f V\n", voltage);
    }

    int internal_N = (Nx - 2) * (Ny - 2);
    if (internal_N <= 0) {
        fprintf(stderr, "Error: Grid is too small. Need at least 1 internal point (Nx, Ny >= 3).\n");
        return 1;
    }

    // --- Write parameters to a file in the current working directory ---
    write_params_to_file(Nx, Ny, h);

    // --- Allocate memory for vectors ---
    Vector phi_solution_internal = allocate_vector(internal_N);
    Vector b_rhs = allocate_vector(internal_N);
    Vector rho_full_grid = allocate_vector(Nx * Ny);
    Vector phi_boundary_full_grid = allocate_vector(Nx * Ny);

    // --- Initialize problem data ---
    initialize_vector(phi_solution_internal, internal_N, 0.0);
    setup_charge_density(rho_full_grid, Nx, Ny);
    setup_boundary_conditions(phi_boundary_full_grid, Nx, Ny, voltage);

    // Create the right-hand side vector 'b' based on charge density and boundary conditions.
    create_b(b_rhs, rho_full_grid, phi_boundary_full_grid, Nx, Ny, h, epsilon_0);

    // --- Solve the linear system using the GCR method ---
    printf("\nStarting GCR solver...\n");
    double start_time = tic();

    int gcr_status = solve_gcr(phi_solution_internal, b_rhs, internal_N,
                               apply_A,
                               rho_full_grid, phi_boundary_full_grid,
                               Nx, Ny, h, epsilon_0,
                               MAX_GCR_ITERATIONS, GCR_TOLERANCE);

    double end_time = toc(start_time);
    printf("GCR solver finished in %.4f seconds.\n", end_time);

    if (gcr_status != -1) {
        printf("GCR converged successfully in %d iterations.\n", gcr_status);
    } else {
        printf("GCR did not converge within the maximum number of iterations.\n");
    }

    // --- Output results to files ---
    write_solution_to_file(phi_solution_internal, phi_boundary_full_grid, Nx, Ny, h, "/content/drive/MyDrive/buw/phi_solution.txt");
    calculate_and_write_electric_field(phi_solution_internal, phi_boundary_full_grid, Nx, Ny, h, "/content/drive/MyDrive/buw/Ex_field.txt", "/content/drive/MyDrive/buw/Ey_field.txt");

    // --- Cleanup: Free all dynamically allocated memory ---
    free_vector(phi_solution_internal);
    free_vector(b_rhs);
    free_vector(rho_full_grid);
    free_vector(phi_boundary_full_grid);

    return 0;
}
