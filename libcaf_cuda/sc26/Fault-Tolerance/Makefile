NVCC ?= nvcc
CXX ?= g++
CXXFLAGS = -std=c++20 -include cstddef -include cerrno -include climits

CUDA_INCLUDE_DIR ?=
CUDA_LIB_DIR ?= 

CAF_INCLUDE_DIR ?=
CAF_LIB_DIR ?=

LIBS = -L$(CAF_LIB_DIR) \
			-lcaf_core \
			-lcaf_io \
			-lcaf_net \
			-lcaf_cuda \
			-L$(CUDA_LIB_DIR) \
			-lcuda \
			-lcublas \
			-lcusparse



INCLUDES = -I$(CAF_INCLUDE_DIR) \
		   -I$(CUDA_INCLUDE_DIR)

LDFLAGS=-Wl,-rpath,$(CAF_LIB_DIR) \
		 		-Wl,-rpath,$(CUDA_LIB_DIR)
NVCC_ARCH ?= sm_$(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')

all: monte_carlo main

monte_carlo: monte_carlo.cu
	$(NVCC) -g -arch=$(NVCC_ARCH) --cubin monte_carlo.cu -o monte_carlo.cubin

main: main.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o main main.cpp $(LIBS) $(LDFLAGS)


clean:
	rm -f monte_carlo.cubin
	rm -f main

.PHONY: all clean
