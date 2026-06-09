NVCC ?= nvcc
CXX ?= g++

CXXFLAGS = -std=c++20 -include cstddef -include cerrno -include climits

CUDA_INCLUDE_DIR ?=
CUDA_LIB_DIR ?=

CAF_INCLUDE_DIR ?=
CAF_LIB_DIR ?=

INCLUDES = \
	-I$(CAF_INCLUDE_DIR) \
	-I$(CUDA_INCLUDE_DIR)

LIBS = \
	-L$(CAF_LIB_DIR) \
	-lcaf_core \
	-lcaf_io \
	-lcaf_net \
	-lcaf_cuda \
	-L$(CUDA_LIB_DIR) \
	-lcuda \
	-lcublas \
	-lcusparse

LDFLAGS = \
	-Wl,-rpath,$(CAF_LIB_DIR) \
	-Wl,-rpath,$(CUDA_LIB_DIR)

NVCC_ARCH ?= sm_$(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')

TARGETS = cuda_native command_runner actor_facade

all: mmul_kernel $(TARGETS)

mmul_kernel: mmul.cu
	$(NVCC) -arch=$(NVCC_ARCH) --cubin $< -o mmul.cubin

cuda_native: cuda_native.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LIBS) $(LDFLAGS)

command_runner: command_runner.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LIBS) $(LDFLAGS)

actor_facade: actor_facade.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LIBS) $(LDFLAGS)

clean:
	rm -f mmul.cubin
	rm -f $(TARGETS)

.PHONY: all clean
