#include <stdio.h>   // For standard input/output functions (printf, fprintf)
#include <stdlib.h>  // For atoi, malloc, free, exit
#include <math.h>    // For mathematical functions (e.g., sqrt, fabs)
#include <omp.h>     // For OpenMP functions like omp_set_num_threads, omp_get_wtime

// Include custom headers for Poisson problem, GCR solver, and utility functions
#include "poisson.h"
#include "gcr_solver.h"
#include "utils.h"

// Define problem constants
#define DEFAULT_NX 102 // Default number of grid points in x-direction (including boundaries)
#define DEFAULT_NY 102 // Default number of grid points in y-direction (including boundaries)
#define H_SPACING 0.01 // Grid spacing (delta x = delta y) in meters
#define EPSILON_0 8.854e-12 // Permittivity of free space (F/m) - a fundamental constant

// Define GCR solver parameters
#define MAX_GCR_ITERATIONS 10000 // Maximum iterations for the GCR solver
#define GCR_TOLERANCE 1e-12    // Relative residual norm tolerance for convergence

/**
 * @brief Sets up the charge density (rho) on the full grid.
 * For this example, a single positive point charge is placed near the center.
 * You can modify this function to define arbitrary charge distributions.
 * @param rho The vector representing the charge density on the full grid (Nx * Ny).
 * @param Nx Total grid points in x-direction.
 * @param Ny Total grid points in y-direction.
 */
void setup_charge_density(Vector rho, int Nx, int Ny) {
    // Initialize all charge density values to zero.
    initialize_vector(rho, Nx * Ny, 0.0);

    // Example: Place a positive point charge at a specific grid location.
    // The charge is placed at (center_y, center_x) in grid coordinates.
    int center_x = Nx / 2;
    int center_y = Ny / 2;
    double charge_value = 1e-9; // Example charge value in Coulombs

    // Assign the charge value to the corresponding grid point.
    // get_index maps the 2D (y,x) grid coordinate to a 1D array index.
    rho[get_index(center_y, center_x, Nx)] = charge_value;

    // You can add more complex charge distributions here, e.g.,
    // a line of charges, a charged plate, multiple point charges, etc.
    // For example, to simulate a charged line along x=Nx/4:
    // for (int i = 1; i < Ny - 1; ++i) {
    //     rho[get_index(i, Nx / 4, Nx)] = 1e-10;
    // }
}

/**
 * @brief Sets up the boundary conditions for the electric potential (phi) on the full grid.
 * For this example, all boundaries are set to zero potential (grounded box).
 * You can modify this function to define different boundary conditions
 * (e.g., constant potential plates, mixed boundary conditions).
 * @param phi_boundary The vector representing the potential on the full grid (Nx * Ny).
 * @param Nx Total grid points in x-direction.
 * @param Ny Total grid points in y-direction.
 */
void setup_boundary_conditions(Vector phi_boundary, int Nx, int Ny) {
    // Initialize all potential values on the full grid to zero.
    // This will serve as the default for boundary points.
    initialize_vector(phi_boundary, Nx * Ny, 0.0);

    // Example: Set specific boundary conditions.
    // For a simple grounded box, all boundary points remain 0.0.
    //
    // To simulate a capacitor (as in 'condensator.m' from the lecture):
    // Set top boundary (i=Ny-1) to +V and bottom boundary (i=0) to -V.
    // All other boundaries (left/right) would typically be 0 or insulated.
    //
    // double V_plate = 10.0; // Example voltage for capacitor plates
    // for (int j = 0; j < Nx; ++j) {
    //     phi_boundary[get_index(0, j, Nx)] = -V_plate;       // Bottom plate
    //     phi_boundary[get_index(Ny - 1, j, Nx)] = V_plate;   // Top plate
    // }
    //
    // For left/right boundaries, if they are also conductors at 0V,
    // they remain 0 from the initialization. If they are insulated,
    // you would use different numerical methods (e.g., Neumann conditions),
    // which are more complex than simple Dirichlet for this project.
}

/**
 * @brief Main function of the GCR-Solve program.
 * Handles program setup, calls the GCR solver, and outputs results.
 * @param argc The number of command-line arguments.
 * @param argv An array of strings containing the command-line arguments.
 * Expected usage: ./electrostatics_solver [Nx] [Ny] [num_threads]
 * @return 0 on successful execution, 1 on error.
 */
int main(int argc, char *argv[]) {
    // Initialize grid dimensions and physical constants with default values.
    int Nx = DEFAULT_NX;
    int Ny = DEFAULT_NY;
    double h = H_SPACING;
    double epsilon_0 = EPSILON_0;

    // Parse command-line arguments for grid dimensions (Nx, Ny).
    // This allows users to easily test different problem sizes.
    if (argc >= 3) {
        Nx = atoi(argv[1]); // Convert string argument to integer for Nx
        Ny = atoi(argv[2]); // Convert string argument to integer for Ny
        // Basic validation for grid size (must be at least 3x3 to have internal points)
        if (Nx < 3 || Ny < 3) {
            fprintf(stderr, "Error: Nx and Ny must be at least 3 (to have internal points).\n");
            return 1; // Exit with error code
        }
        printf("Using grid size: Nx=%d, Ny=%d\n", Nx, Ny);
    } else {
        printf("Using default grid size: Nx=%d, Ny=%d\n", Nx, Ny);
        printf("Usage: %s [Nx] [Ny] [num_threads]\n", argv[0]);
    }

    // Parse command-line argument for the number of OpenMP threads.
    // This is crucial for strong scaling analysis.
    if (argc == 4) {
        int num_threads = atoi(argv[3]); // Convert string argument to integer for num_threads
        if (num_threads > 0) {
            omp_set_num_threads(num_threads); // Set the number of threads for OpenMP
            printf("Set OpenMP threads to: %d\n", num_threads);
        } else {
            fprintf(stderr, "Error: Number of threads must be positive.\n");
            return 1; // Exit with error code
        }
    } else {
        // If no thread count is specified, OpenMP uses its default (often number of CPU cores).
        printf("Using default OpenMP threads (usually number of CPU cores).\n");
    }

    // Calculate the number of internal grid points.
    // The GCR solver operates only on these points.
    int internal_Nx = Nx - 2;
    int internal_Ny = Ny - 2;
    int internal_N = internal_Nx * internal_Ny;

    // Validate that there's at least one internal point.
    if (internal_N <= 0) {
        fprintf(stderr, "Error: Grid is too small. Need at least 1 internal point (Nx, Ny >= 3).\n");
        return 1; // Exit with error code
    }

    // --- Allocate memory for vectors ---
    // phi_solution_internal: Stores the computed electric potential for internal points.
    Vector phi_solution_internal = allocate_vector(internal_N);
    // b_rhs: Stores the right-hand side vector of the linear system A*y = b.
    Vector b_rhs = allocate_vector(internal_N);

    // rho_full_grid: Stores the charge density for the entire grid (including boundaries).
    //                Boundary values are typically zero for charge density.
    Vector rho_full_grid = allocate_vector(Nx * Ny);
    // phi_boundary_full_grid: Stores the fixed potential values on the boundaries.
    //                         Internal points in this vector are irrelevant for the solver.
    Vector phi_boundary_full_grid = allocate_vector(Nx * Ny);

    // --- Initialize problem data ---
    // Set initial guess for phi (internal points) to zeros.
    initialize_vector(phi_solution_internal, internal_N, 0.0);

    // Set up the charge density distribution.
    setup_charge_density(rho_full_grid, Nx, Ny);
    // Set up the boundary conditions for the potential.
    setup_boundary_conditions(phi_boundary_full_grid, Nx, Ny);

    // Create the right-hand side vector 'b' based on charge density and boundary conditions.
    create_b(b_rhs, rho_full_grid, phi_boundary_full_grid, Nx, Ny, h, epsilon_0);

    // --- Solve the linear system using the GCR method ---
    printf("\nStarting GCR solver...\n");
    double start_time = tic(); // Start timing the solver execution

    // Call the GCR solver. It will iteratively refine 'phi_solution_internal'.
    int gcr_status = solve_gcr(phi_solution_internal, b_rhs, internal_N,
                               apply_A, // Function pointer to apply the A matrix
                               rho_full_grid, phi_boundary_full_grid, // Parameters for apply_A
                               Nx, Ny, h, epsilon_0,                   // Parameters for apply_A
                               MAX_GCR_ITERATIONS, GCR_TOLERANCE);

    double end_time = toc(start_time); // Stop timing
    printf("GCR solver finished in %.4f seconds.\n", end_time);

    // Report solver convergence status.
    if (gcr_status == 0) {
        printf("GCR converged successfully.\n");
    } else {
        printf("GCR did not converge within the maximum number of iterations.\n");
    }

    // --- Output results to files ---
    // Write the final potential solution to a file.
    // The output file will be in the 'data/' directory on your host machine.
    write_solution_to_file(phi_solution_internal, phi_boundary_full_grid, Nx, Ny, h, "data/phi_solution.txt");
    // Calculate and write the electric field components (Ex, Ey) to files.
    // These will also be in the 'data/' directory.
    calculate_and_write_electric_field(phi_solution_internal, phi_boundary_full_grid, Nx, Ny, h, "data/Ex_field.txt", "data/Ey_field.txt");

    // --- Cleanup: Free all dynamically allocated memory ---
    free_vector(phi_solution_internal);
    free_vector(b_rhs);
    free_vector(rho_full_grid);
    free_vector(phi_boundary_full_grid);

    return 0; // Indicate successful program execution
}
