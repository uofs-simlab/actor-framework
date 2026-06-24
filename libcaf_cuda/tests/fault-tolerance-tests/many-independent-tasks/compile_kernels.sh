#!/bin/bash
set -e

# Detect first GPU compute capability to ensure binary compatibility
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -n1)
SM_ARCH="sm_${ARCH/./}"
echo "Using NVCC arch flag: $SM_ARCH"

# Compile kernels to cubin for Driver API loading
nvcc -arch=$SM_ARCH -cubin mmul.cu -o mmul.cubin
nvcc -arch=$SM_ARCH -cubin vector_add.cu -o vector_add.cubin
nvcc -arch=$SM_ARCH -cubin conv1d.cu -o conv1d.cubin
nvcc -arch=$SM_ARCH -cubin poison.cu -o poison.cubin

echo "Kernels compiled successfully for $SM_ARCH."
