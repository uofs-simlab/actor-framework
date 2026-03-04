cmake_minimum_required(VERSION 3.10)
project(cuda_driver_mmul LANGUAGES CXX)

# Find nvcc
find_program(NVCC_EXECUTABLE nvcc)
if(NOT NVCC_EXECUTABLE)
    message(FATAL_ERROR "nvcc not found. Make sure CUDA toolkit is installed and nvcc is in PATH.")
endif()

# Try to locate CUDA driver library (libcuda)
find_library(CUDA_DRIVER_LIB NAMES cuda HINTS ENV CUDA_HOME /usr/local/cuda/lib64 /usr/local/cuda/lib)
if(NOT CUDA_DRIVER_LIB)
    message(FATAL_ERROR "Could not find CUDA driver library (libcuda). Ensure CUDA is installed.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ensure we build PTX from mmul.cu into build directory as mmul.ptx
set(MMUL_CU "${CMAKE_SOURCE_DIR}/mmul.cu")
set(MMUL_PTX "${CMAKE_BINARY_DIR}/mmul.ptx")

add_custom_command(
    OUTPUT "${MMUL_PTX}"
    COMMAND ${NVCC_EXECUTABLE} -ptx ${MMUL_CU} -o "${MMUL_PTX}"
    DEPENDS "${MMUL_CU}"
    COMMENT "Compiling mmul.cu -> mmul.ptx"
    VERBATIM
)

add_custom_target(kernel_ptx ALL DEPENDS "${MMUL_PTX}")

# Host executable
add_executable(matrix_mul_driver matrix_mul_driver.cpp)
add_dependencies(matrix_mul_driver kernel_ptx)

# Make sure PTX is copied to the executable directory so the program can find it at runtime
add_custom_command(TARGET matrix_mul_driver POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${MMUL_PTX}"
        $<TARGET_FILE_DIR:matrix_mul_driver>
)

# Link to the CUDA driver library (libcuda)
target_link_libraries(matrix_mul_driver PRIVATE ${CUDA_DRIVER_LIB})

# Include path - try to find cuda headers (optional)
find_path(CUDA_INCLUDE_DIR cuda.h HINTS ENV CUDA_HOME /usr/local/cuda/include)
if(CUDA_INCLUDE_DIR)
    target_include_directories(matrix_mul_driver PRIVATE ${CUDA_INCLUDE_DIR})
endif()
