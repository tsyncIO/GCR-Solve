#include <stdio.h>
#include <stdlib.h>
#include <omp.h> // for omp_get_wtime

// User-defined type alias for a vector
typedef double* Vector;

// Dummy versions of external functions for this self-contained snippet
Vector allocate_vector(int n) { return (Vector)malloc(n * sizeof(double)); }
void free_vector(Vector v) { if (v != NULL) free(v); }
void initialize_vector(Vector v, int n, double val) { for (int i = 0; i < n; ++i) v[i] = val; }
int get_index(int i, int j, int Nx) { return i * Nx + j; }
double tic() { return omp_get_wtime(); }
double toc(double start_time) { return omp_get_wtime() - start_time; }
void setup_charge_density(Vector rho, int Nx, int Ny) {
    initialize_vector(rho, Nx * Ny, 0.0);
    // Add a simplified charge for demonstration
    rho[get_index(Ny / 2, Nx / 2, Nx)] = 1.0;
}
void setup_boundary_conditions(Vector phi_boundary, int Nx, int Ny) {
    initialize_vector(phi_boundary, Nx * Ny, 0.0);
}
void create_b(Vector b_rhs, const Vector rho, const Vector phi_boundary, int Nx, int Ny, double h, double epsilon_0) {
    // This is a simplified stand-in for the actual create_b function
    printf("Simulating creation of right-hand side vector 'b'.\n");
    for (int i = 0; i < (Nx-2)*(Ny-2); ++i) {
        b_rhs[i] = (rho[get_index(i + 1, (i % (Nx - 2)) + 1, Nx)] / epsilon_0) * h * h;
    }
}
int solve_gcr(Vector y, const Vector b, int internal_N, void (*apply_A_func)(), const Vector rho, const Vector phi_boundary, int Nx, int Ny, double h, double epsilon_0, int max_iter, double tol) {
    printf("Simulating GCR solver call...\n");
    // Just a placeholder, the real solver would go here
    return 0; // 0 for success
}

// Updated function to write solution to a file with header
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
                phi_val = solution_phi[(i - 1) * (Nx - 2) + (j - 1)];
            }
            fprintf(fp, "%.6f ", phi_val);
        }
        fprintf(fp, "\n");
    }

    printf("Writing solution to %s\n", filename);
    fclose(fp);
}

// Updated function to write E-field to a file with header
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

    printf("Simulating writing electric field to %s and %s\n", filename_Ex, filename_Ey);

    // Dummy logic to simulate writing some E-field data
    for (int i = 0; i < Ny * Nx; ++i) {
        fprintf(fp_Ex, "%.6f ", (double)i);
        fprintf(fp_Ey, "%.6f ", (double)i);
    }

    fclose(fp_Ex);
    fclose(fp_Ey);
}


int main() {
    // We'll hardcode values instead of parsing arguments
    int Nx = 10, Ny = 10;
    double h = 0.1;
    
    int internal_N = (Nx - 2) * (Ny - 2);

    printf("Using grid size: Nx=%d, Ny=%d with spacing h=%f\n", Nx, Ny, h);

    // --- Allocate memory ---
    Vector phi_solution_internal = allocate_vector(internal_N);
    Vector b_rhs = allocate_vector(internal_N);
    Vector rho_full_grid = allocate_vector(Nx * Ny);
    Vector phi_boundary_full_grid = allocate_vector(Nx * Ny);

    // --- Initialize problem data ---
    initialize_vector(phi_solution_internal, internal_N, 0.0);
    setup_charge_density(rho_full_grid, Nx, Ny);
    setup_boundary_conditions(phi_boundary_full_grid, Nx, Ny);
    
    // Create the right-hand side vector
    create_b(b_rhs, rho_full_grid, phi_boundary_full_grid, Nx, Ny, 0.01, 8.854e-12);

    // --- Solve and time ---
    double start_time = tic();
    // The actual GCR solver would be called here
    int status = solve_gcr(phi_solution_internal, b_rhs, internal_N, NULL, NULL, NULL, Nx, Ny, 0.01, 8.854e-12, 1000, 1e-6);
    double end_time = toc(start_time);
    
    printf("\nMain program finished in %.4f seconds.\n", end_time);
    if (status == 0) {
        printf("Solver finished successfully.\n");
    } else {
        printf("Solver failed to converge.\n");
    }

    // --- Output results ---
    write_solution_to_file(phi_solution_internal, phi_boundary_full_grid, Nx, Ny, h, "/content/drive/MyDrive/buw/phi_solution.txt");
    calculate_and_write_electric_field(phi_solution_internal, phi_boundary_full_grid, Nx, Ny, h, "/content/drive/MyDrive/buw/Ex_field.txt", "/content/drive/MyDrive/buw/Ey_field.txt");
    
    // --- Cleanup ---
    free_vector(phi_solution_internal);
    free_vector(b_rhs);
    free_vector(rho_full_grid);
    free_vector(phi_boundary_full_grid);
    
    return 0;
}

