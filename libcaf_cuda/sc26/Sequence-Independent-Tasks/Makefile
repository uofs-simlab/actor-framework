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
			-lcuda

INCLUDES = -I$(CAF_INCLUDE_DIR) \
		  	 	 -I$(CUDA_INCLUDE_DIR)

LDFLAGS=-Wl,-rpath,$(CAF_LIB_DIR) \
		 		-Wl,-rpath,$(CUDA_LIB_DIR)
NVCC_ARCH ?= sm_$(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')

all: mmul_kernel base_cuda_program actor_facade actor_command_runner

mmul_kernel: mmul.cu
	$(NVCC) -g -arch=$(NVCC_ARCH) --cubin mmul.cu -o mmul.cubin

base_cuda_program: cuda_native.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o main_cuda_native cuda_native.cpp $(LIBS) $(LDFLAGS)

actor_facade: actor_facade.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o main_actor_facade actor_facade.cpp $(LIBS) $(LDFLAGS)

actor_command_runner: command_runner.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o main_command_runner command_runner.cpp $(LIBS) $(LDFLAGS)

clean:
	rm -f mmul.cubin
	rm -f main_cuda_native
	rm -f main_actor_facade
	rm -f main_command_runner