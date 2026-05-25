cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(CUDALaplacianBenchmark LANGUAGES CXX CUDA)

# Set the C++ standard required by your code and standard libraries
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Request standard compliant CUDA compilation flags
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Optimization flags for benchmarking (Release mode)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Find the CUDA Toolkit libraries (cuBLAS and cuSPARSE)
find_package(CUDAToolkit REQUIRED)

# Define the executable target
# Replace 'main.cu' with whatever you name your source code file
add_executable(cuda_benchmark main.cu)

# Target compile features/options (if required)
target_compile_options(cuda_benchmark PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
)

# Link the required NVIDIA hardware-accelerated libraries
target_link_libraries(cuda_benchmark PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cusparse
)

# (Optional) Automatically match target GPU architecture if you run it locally
# If compiling for a specific architecture, you can uncomment and adjust the line below:
# set_target_properties(cuda_benchmark PROPERTIES CUDA_ARCHITECTURES "75;80;86")
