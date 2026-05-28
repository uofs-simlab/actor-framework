#project(caf_opencl C CXX)
include(CMakePrintHelpers)

# Get the cuda related dependencies first so we have the nvcc path
find_package(CUDAToolkit REQUIRED)

# Set the GPU architecture. 'native' detects the local GPU capability.
# You can override this with -DCAF_CUDA_ARCH=sm_80 for example.
set(CAF_CUDA_ARCH "native" CACHE STRING "Target CUDA architecture (e.g., sm_70, native)")

# Find all .cu files recursively in the caf/ directory relative to this CMakeLists.txt
file(GLOB_RECURSE CAF_CUDA_KERNELS "${CMAKE_CURRENT_SOURCE_DIR}/caf/*.cu")

foreach(CU_FILE ${CAF_CUDA_KERNELS})
    # Get the base name of the .cu file (e.g., "some_kernel")
    get_filename_component(BASENAME ${CU_FILE} NAME_WE)
    # Get the directory of the .cu file (e.g., "/path/to/libcaf_cuda/caf/cuda")
    get_filename_component(DIRNAME ${CU_FILE} DIRECTORY)
    
    # Construct the output .cubin file path (e.g., "/path/to/libcaf_cuda/caf/cuda/some_kernel.cubin")
    set(CUBIN_FILE "${DIRNAME}/${BASENAME}.cubin")

    # Create a unique target name based on the relative path to avoid collisions
    file(RELATIVE_PATH REL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/caf" ${CU_FILE})
    string(MAKE_C_IDENTIFIER "cubin_${REL_PATH}" TARGET_NAME)

    # Add a custom command to compile the .cu file into a .cubin
    add_custom_command(
        OUTPUT ${CUBIN_FILE}
        COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
            -cubin
            -arch=${CAF_CUDA_ARCH}
            -o ${CUBIN_FILE}
            ${CU_FILE}
        DEPENDS ${CU_FILE}
        COMMENT "Compiling CUDA kernel ${CU_FILE} to ${CUBIN_FILE}"
        VERBATIM
    )
    
    # Add a custom target to ensure the cubin is built.
    add_custom_target(${TARGET_NAME} ALL DEPENDS ${CUBIN_FILE})
endforeach()

# get header files; only needed by CMake generators,
# e.g., for creating proper Xcode projects
file(GLOB_RECURSE CAF_CUDA_HEADERS "caf/*.hpp")

#cmake_print_variables(CAF_CUDA_HEADERS)



# list cpp files excluding platform-dependent files
file(GLOB_RECURSE LIBCAF_CUDA_SRCS
    src/*.cpp
)

# -- add targets ---------------------------------------------------------------

caf_add_component(
  cuda
  DEPENDENCIES
    PUBLIC
    CAF::core
      $<$<CXX_COMPILER_ID:MSVC>:ws2_32>
    PRIVATE
      CAF::internal
      #CAF::CORE
      CUDA::nvrtc
      CUDA::cuda_driver
    HEADERS
    ${CAF_CUDA_HEADERS}
  SOURCES
    ${LIBCAF_CUDA_SRCS}
     )
