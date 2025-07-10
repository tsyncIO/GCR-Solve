GCR-Solve: Electrostatics Solver

Project Overview

GCR-Solve is a high-performance C application designed to solve the Poisson equation using the Generalized Conjugate Residual (GCR) iterative method. This project is developed with OpenMP for CPU-based parallelization and is fully containerized using Docker, ensuring a consistent and reproducible development and execution environment.



This project was undertaken as a part of my Master's program, M.Sc. in CSiS (Computational Science in Engineering and Simulation), during the Summer 2025 semester at the University of Wuppertal. It is a term paper project for the course "Numerical Methods in Classical Field Theory and Quantum Mechanics (C0200)", conducted under the supervision of Dr. Tomasz Korzec from the High-performance computing in theoretical physics research group.



The Poisson equation is fundamental in electrostatics, describing how electric potential is distributed in space given a charge density and permittivity. This solver discretizes the problem using the finite difference method, transforming it into a large sparse linear system, which is then efficiently solved by the GCR algorithm without explicitly constructing the matrix.



Features

Poisson Equation Solver: Solves for electric potential in a 2D rectangular domain.



Generalized Conjugate Residual (GCR) Method: An iterative Krylov subspace method optimized for sparse linear systems.



Implicit Matrix-Vector Product: The matrix is never explicitly formed, saving significant memory and computation for large grids.



OpenMP Parallelization: Leverages multi-core CPUs for accelerated computation, particularly in matrix-vector products and vector operations.



Docker Containerization: Provides a self-contained, reproducible environment with all necessary compilers and libraries (GCC, OpenMP, Valgrind).



Output to ASCII Files: Generates potential and electric field data in a format suitable for external plotting tools (e.g., MATLAB, Python with Matplotlib).



Memory Debugging: Includes Valgrind in the Docker image for robust memory error detection.



Prerequisites

Before you begin, ensure you have the following installed on your system:



Git: For cloning the repository.



Download Git



Docker Desktop: Includes Docker Engine and Docker Compose.



Download Docker Desktop (for Windows, macOS, Linux)



Getting Started

Follow these steps to set up and run the GCR-Solve project.



Clone the Repository:

Open your terminal (PowerShell on Windows, or Bash on Linux/macOS) and clone the project:



git clone https://github.com/YOUR\_USERNAME/GCR-Solve.git

cd GCR-Solve







(Remember to replace YOUR\_USERNAME with your actual GitHub username)



Build the Docker Image:

This command will build the Docker image based on the Dockerfile. This might take a few minutes the first time as it downloads Ubuntu and installs compilers.



docker-compose build







Start the Development Container:

This command launches a new container instance and drops you into a bash shell inside it. Your local src/ and data/ directories will be mounted into the container.



docker-compose run --rm gcr-solve-dev







You will see a prompt like root@<container\_id>:/app#.



Compilation and Execution (Inside the Docker Container)

Once you are inside the Docker container's terminal:



Navigate to the Source Directory:



cd src







Compile the C Code:

This command compiles all source files (.c) and links them, creating the executable electrostatics\_solver. The -O3 flag enables high optimization, and -fopenmp enables OpenMP parallelism.



gcc -Wall -Wextra -O3 -fopenmp main.c poisson.c gcr\_solver.c utils.c -o electrostatics\_solver -lm







Run the Solver:

The executable takes optional command-line arguments for grid dimensions (Nx, Ny), the number of OpenMP threads (num\_threads), and a maximum number of GCR iterations (max\_iterations).



Basic Usage:



./electrostatics\_solver \[Nx] \[Ny] \[num\_threads] \[max\_iterations]







Examples:



Run with default settings (102x102 grid, default OpenMP threads, max 10000 iterations):



./electrostatics\_solver







Run with a 102x102 grid, 4 OpenMP threads, max 10000 iterations:



export OMP\_NUM\_THREADS=4 # Set OpenMP threads via environment variable

./electrostatics\_solver 102 102







Run with a 502x502 grid, 8 OpenMP threads, max 5000 iterations:



export OMP\_NUM\_THREADS=8

./electrostatics\_solver 502 502 8 5000







Run sequentially (1 thread) for a 502x502 grid, max 10000 iterations:



export OMP\_NUM\_THREADS=1

./electrostatics\_solver 502 502







The program will print iteration progress and final convergence status/time to the console.



Understanding the Output Files

After running the solver, output files will be generated in the data/ directory (which is mounted to your host machine).



data/phi\_solution.txt: Contains the computed electric potential values for the entire grid (including boundary points), formatted as a 2D matrix.



data/Ex\_field.txt: Contains the X-component of the electric field for the entire grid.



data/Ey\_field.txt: Contains the Y-component of the electric field for the entire grid.



These files can be loaded and visualized using tools like MATLAB, GNU Octave, or Python with libraries like NumPy and Matplotlib.



Example Python Plotting Script (on your host machine, after running the solver):



import numpy as np

import matplotlib.pyplot as plt



\# Load data from the 'data' directory

try:

    phi\_data = np.loadtxt('data/phi\_solution.txt')

    ex\_data = np.loadtxt('data/Ex\_field.txt')

    ey\_data = np.loadtxt('data/Ey\_field.txt')

except IOError:

    print("Error: Make sure the solver has run and generated files in the 'data/' directory.")

    exit()



\# Plot Electric Potential (phi)

plt.figure(figsize=(8, 6))

plt.imshow(phi\_data, cmap='viridis', origin='lower')

plt.colorbar(label='Electric Potential (phi)')

plt.title('Electric Potential Field')

plt.xlabel('X-grid index')

plt.ylabel('Y-grid index')

plt.tight\_layout()

plt.show()



\# Plot Electric Field (E) as a quiver plot

\# Subsample for clearer visualization on large grids

step = max(1, phi\_data.shape\[0] // 20) # Adjust subsampling step

Y, X = np.mgrid\[0:phi\_data.shape\[0]:step, 0:phi\_data.shape\[1]:step]



plt.figure(figsize=(10, 8))

plt.imshow(phi\_data, cmap='viridis', origin='lower', alpha=0.6) # Background potential

plt.colorbar(label='Electric Potential (phi)')

plt.quiver(X, Y, ex\_data\[::step, ::step], ey\_data\[::step, ::step], color='white', scale=np.max(np.sqrt(ex\_data\*\*2 + ey\_data\*\*2))\*2, width=0.002)

plt.title('Electric Field (Vector Plot)')

plt.xlabel('X-grid index')

plt.ylabel('Y-grid index')

plt.tight\_layout()

plt.show()



Parallelization and Performance Analysis

The project is designed to demonstrate OpenMP's efficiency. You can perform strong scaling analysis by:



Choosing a fixed, sufficiently large problem size (e.g., 502 502).



Running the solver with varying numbers of OpenMP threads (e.g., 1, 2, 4, 8, up to your CPU's logical core count).



Recording the execution time (printed to console by the solver).



Calculating Speedup: Speedup = Time\_sequential (1 thread) / Time\_parallel (N threads).



Plotting Speedup vs. Number of Threads to visualize the parallel performance.



Memory Debugging with Valgrind

The Docker image includes Valgrind, a powerful tool for detecting memory errors (leaks, invalid reads/writes, etc.). To run your solver with Valgrind:



\# Inside the container, in the src directory

valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./electrostatics\_solver 102 102 4 100



(Adjust grid size, threads, and iterations for a quicker Valgrind run, as it adds significant overhead).



Valgrind will output a detailed report on any memory issues it finds.



Project Structure

GCR-Solve/

├── Dockerfile              # Defines the Docker image build process

├── docker-compose.yml      # Orchestrates the Docker container setup

├── .gitignore              # Specifies files/directories to ignore in Git

├── README.md               # This file

├── src/                    # Contains all C source code files

│   ├── main.c              # Main program logic, setup, and solver call

│   ├── poisson.c           # Implementation of Poisson-related functions (apply\_A, create\_b)

│   ├── poisson.h           # Header for poisson.c

│   ├── gcr\_solver.c        # Implementation of GCR algorithm and vector operations

│   ├── gcr\_solver.h        # Header for gcr\_solver.c

│   ├── utils.c             # Implementation of utility functions (memory, file I/O, timing)

│   └── utils.h             # Header for utils.c

└── data/                   # Directory for output files (mounted from host)

    └── (output files like phi\_solution.txt, Ex\_field.txt, Ey\_field.txt)



License

(Optional: Add a license section here, e.g., MIT License. You can choose one from choosealicense.com.)

