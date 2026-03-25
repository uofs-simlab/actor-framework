# Makefile for example_3
NVCC=/usr/local/cuda-12.2/bin/nvcc

LIBS=-L/u1/users/kck540/actors/cuda-caf/lib -lcaf_core -lcaf_io -lcaf_net -lcaf_cuda -lcuda
INCLUDES=-I/u1/users/kck540/actors/cuda-caf/include -I/usr/local/cuda-12.2/targets/x86_64-linux/include


all: cuda actors

cuda:
	$(NVCC) --cubin matmul_kernel.cu -o matmul.cubin -arch=sm_61

actors:
	g++ main.cpp -std=c++20 $(INCLUDES) $(LIBS) -o main

clean:
	rm -f main matmul.cubin

.PHONY: cuda actors
