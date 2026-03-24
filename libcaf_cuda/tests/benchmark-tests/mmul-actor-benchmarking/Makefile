NVCC=/home/kklenk/miniforge3/envs/research-env/bin/nvcc -Wno-deprecated-gpu-targets
CXX=/home/kklenk/miniforge3/envs/research-env/bin/g++
CXXFLAGS=-std=c++20 -include cstddef -include cerrno -include climits 

LIBS=-L/home/kklenk/miniforge3/envs/research-env/lib \
		 -L/home/kklenk/.local/caf/lib \
		 -lcaf_core -lcaf_io -lcaf_net -lcaf_cuda -lcuda

CUDA_INCLUDES=-I/home/kklenk/miniforge3/envs/research-env/targets/x86_64-linux/include
INCLUDES=-I/home/kklenk/.local/caf/include \
		 -I/home/kklenk/miniforge3/envs/research-env/include \
		 $(CUDA_INCLUDES)

LDFLAGS=-Wl,-rpath,/home/kklenk/.local/caf/lib \
		 -Wl,-rpath,/opt/cuda/targets/x86_64-linux/lib
NVCC_ARCH := sm_$(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')

.PHONY: all clean

all: mmul.cubin main

mmul.cubin: mmul.cu
	@echo "[NVCC] Building example_1 cubin for $(NVCC_ARCH)"
	$(NVCC) -g -arch=$(NVCC_ARCH) --cubin $< -o $@


main: main.test.cpp
	$(CXX) -g $< $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) $(LIBS) -o $@


clean:
	rm -f mmul.cubin main
