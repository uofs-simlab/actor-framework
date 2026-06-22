cmake_minimum_required(VERSION 3.16.3)

# 1) Enforce C++20 (needs modern C++ features)
set(CMAKE_CXX_STANDARD        20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS      OFF)

# 2) Set CAF source and build directories
set(CAF_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../actor-framework")
set(CAF_BUILD "${CAF_SRC}/build")


#required since cmake cant seem to find the compiler 
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_C_COMPILER "/usr/bin/gcc")

# Enable CUDA as a first-class language for the project
project(CUDA_ACTORS LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(
  "${CAF_SRC}/include"
  "${CAF_SRC}/libcaf_io"
  "${CAF_SRC}/libcaf_core"
  "${CAF_SRC}/libcaf_opencl"
  "${CAF_BUILD}/libcaf_core"  # for generated build_config.hpp  
  "${CAF_SRC}/libcaf_cuda"
)

# --- CUDA Kernel Compilation ---
set(STABILITY_KERNEL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/stability_kernels.cu")
set(STABILITY_KERNEL_CUBIN "${CMAKE_CURRENT_BINARY_DIR}/stability_kernels.cubin")

# Target CUDA architecture. 'native' targets the current machine's GPU (requires CUDA 11.6+).
# You can override this with -DCUDA_ARCH=sm_XX if needed.
set(CUDA_ARCH "native" CACHE STRING "Target CUDA architecture (e.g., native, sm_70, sm_75, sm_80, sm_86)")

add_custom_command(
  OUTPUT ${STABILITY_KERNEL_CUBIN}
  COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
          -cubin
          -arch=${CUDA_ARCH}
          -o ${STABILITY_KERNEL_CUBIN}
          ${STABILITY_KERNEL_SRC}
  DEPENDS ${STABILITY_KERNEL_SRC}
  COMMENT "Compiling CUDA kernel ${STABILITY_KERNEL_SRC} for architecture: ${CUDA_ARCH}"
  VERBATIM
)

set(JACOBI_KERNEL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/jacobi_kernels.cu")
set(JACOBI_KERNEL_CUBIN "${CMAKE_CURRENT_BINARY_DIR}/jacobi_kernels.cubin")

add_custom_command(
  OUTPUT ${JACOBI_KERNEL_CUBIN}
  COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
          -cubin
          -arch=${CUDA_ARCH}
          -o ${JACOBI_KERNEL_CUBIN}
          ${JACOBI_KERNEL_SRC}
  DEPENDS ${JACOBI_KERNEL_SRC}
  COMMENT "Compiling CUDA kernel ${JACOBI_KERNEL_SRC} for architecture: ${CUDA_ARCH}"
  VERBATIM
)

add_custom_target(stability_kernels_cubin ALL DEPENDS ${STABILITY_KERNEL_CUBIN})
add_custom_target(jacobi_kernels_cubin ALL DEPENDS ${JACOBI_KERNEL_CUBIN})


# 5) Declare your executable and its source files
add_executable(test main.test.cpp sparse_utils.cpp)
target_sources(test PRIVATE
  atoms.hpp
  ft_cg_actor.hpp
  ft_cg_jacobi_actor.hpp
  supervisor_actor.hpp
)

target_compile_definitions(test PRIVATE CAF_ENABLE_LOGGING)

target_link_libraries(test
  PRIVATE
    "${CAF_BUILD}/libcaf_core/libcaf_core.so"
    "${CAF_BUILD}/libcaf_io/libcaf_io.so"
    "${CAF_BUILD}/libcaf_cuda/libcaf_cuda.so"
    CUDA::nvrtc
    CUDA::cublas
    CUDA::cusparse
)


# # 5) Declare your executable
# add_executable(hot-potatoe hot-potatoe.cpp sparse_utils.cpp)

# target_compile_definitions(hot-potatoe PRIVATE CAF_ENABLE_LOGGING)

# target_link_libraries(hot-potatoe
#   PRIVATE
#     "${CAF_BUILD}/libcaf_core/libcaf_core.so"
#     "${CAF_BUILD}/libcaf_io/libcaf_io.so"
#     "${CAF_BUILD}/libcaf_cuda/libcaf_cuda.so"
#     CUDA::nvrtc
#     CUDA::cublas
#     CUDA::cusparse
# )


# FindThreads is required for std::thread in the native version
find_package(Threads REQUIRED)

# 6) Declare the native benchmark executable (raw CUDA/cuBLAS/cuSPARSE)
add_executable(workload-native main.native.cpp sparse_utils.cpp native_utils.cu)

target_link_libraries(workload-native
  PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cusparse
    Threads::Threads
)

# 7) Declare the native sorted benchmark executable
add_executable(workload-native-sorted main.native_sorted.cpp sparse_utils.cpp native_utils.cu)

target_link_libraries(workload-native-sorted
  PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cusparse
    Threads::Threads
)
