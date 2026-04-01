#! /bin/bash
set -e

LIBS="-L/u1/users/kck540/actors/cuda-caf/lib -lcaf_core -lcaf_cuda -lcuda"
INCLUDES="-I/u1/users/kck540/actors/cuda-caf/include -I/usr/local/cuda-12.2/targets/x86_64-linux/include"

nvcc=/usr/local/cuda-12.2/bin/nvcc

# Build the CAF example (existing)
g++ main.cpp -std=c++20 $INCLUDES $LIBS -o main

# # Build and run the CUDA matrix-multiply example
# if [ -x "$nvcc" ]; then
#   echo "Building CUDA example..."
#   "$nvcc" -std=c++17 -O2 matmul_example.cu -o matmul_example
#   echo "Running CUDA example..."
#   ./matmul_example 256
# else
#   echo "WARNING: nvcc not found at $nvcc; skipping CUDA example build" >&2
# fi