#!/bin/bash
set -e

# Detect first GPU compute capability
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -n1)
SM_ARCH="sm_${ARCH/./}"
echo "Using NVCC arch flag: $SM_ARCH"

# Compile mmul.cu to cubin in current directory
nvcc -arch=$SM_ARCH -cubin vector_add.cu -o vector_add.cubin
echo "Generated mmul.cubin"
echo "All kernels compiled successfully!"

