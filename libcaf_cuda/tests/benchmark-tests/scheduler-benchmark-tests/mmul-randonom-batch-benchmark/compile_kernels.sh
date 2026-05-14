#!/bin/bash
# Compile kernels to cubin for Driver API loading
nvcc -cubin mmul.cu -o ../mmul.cubin
nvcc -cubin vector_add.cu -o ../vector_add.cubin
nvcc -cubin conv1d.cu -o ../conv1d.cubin
echo "Kernels compiled successfully."