# Use a base image with a recent Ubuntu version
FROM ubuntu:22.04

# Set environment variables for non-interactive installation
# This prevents apt-get from asking questions during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update package list and install necessary packages
# build-essential: Provides essential development tools like gcc, g++, make.
# libomp-dev: OpenMP development files. While GCC often includes OpenMP runtime,
#             this ensures all necessary headers/libs are present.
# valgrind: A powerful memory debugging and profiling tool. Crucial for C projects.
# git: Version control system, useful for development.
# python3, python3-pip: For optional post-processing and plotting scripts (e.g., with matplotlib).
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    libomp-dev \
    valgrind \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* # Clean up apt cache to keep image size down

# Install common Python libraries for numerical analysis and plotting
# matplotlib: For creating plots of your solution and electric fields.
# numpy: For numerical operations in Python scripts.
RUN pip3 install matplotlib numpy

# Set the working directory inside the container
# All subsequent commands (COPY, RUN, CMD) will be relative to this directory.
WORKDIR /app

# Create directories for source code and output data within the container.
# These directories will be mounted from your host machine via docker-compose.
RUN mkdir -p /app/src /app/data

# Default command when the container starts.
# This keeps the container running in a bash shell, allowing you to
# manually compile, run, and debug your C code inside the container.
CMD ["bash"]
