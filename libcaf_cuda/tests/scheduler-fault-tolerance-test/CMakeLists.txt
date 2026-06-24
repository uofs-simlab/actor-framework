cmake_minimum_required(VERSION 3.16) # Back to your original minimum

# Enforce C++20
set(CMAKE_CXX_STANDARD        20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS      OFF)

set(CAF_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../../../actor-framework")
set(CAF_BUILD "${CAF_SRC}/build")

set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_C_COMPILER "/usr/bin/gcc")

project(CUDA_ACTORS)

# Use the old reliable finder to discover where the driver library is hidden
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(
  "${CAF_SRC}/include"
  "${CAF_SRC}/libcaf_io"
  "${CAF_SRC}/libcaf_core"
  "${CAF_SRC}/libcaf_opencl"
  "${CAF_BUILD}/libcaf_core"  
  "${CAF_SRC}/libcaf_cuda"
)

add_executable(scheduler_test scheduler_integration_test.cpp)
target_compile_definitions(scheduler_test PRIVATE CAF_ENABLE_LOGGING)

# --- THE FIX FOR BASELINE ---
add_executable(cuda-baseline main.native.cpp)
target_link_libraries(cuda-baseline PRIVATE 
    CUDA::cudart           # Fixes Runtime API (cudaMalloc, etc.)
    CUDA::cuda_driver   # Fixes Driver API (cuInit, cuModuleLoad) via raw path variable
)

target_link_libraries(scheduler_test
  PRIVATE
    "${CAF_BUILD}/libcaf_core/libcaf_core.so"
    "${CAF_BUILD}/libcaf_io/libcaf_io.so"
    "${CAF_BUILD}/libcaf_cuda/libcaf_cuda.so"
    CUDA::nvrtc 
    CUDA::cublas
    CUDA::cusparse
)