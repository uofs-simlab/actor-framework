#! /bin/bash

export NVCC=/usr/local/cuda-12.2/bin/nvcc

export CUDA_INCLUDE_DIR=/usr/local/cuda-12.2/targets/x86_64-linux/include
export CUDA_LIB_DIR=/usr/local/cuda-12.2/targets/x86_64-linux/lib

export CAF_INCLUDE_DIR=/home/kck540/Projects/cuda-actors/caf/include
export CAF_LIB_DIR=/home/kck540/Projects/cuda-actors/caf/lib

make
