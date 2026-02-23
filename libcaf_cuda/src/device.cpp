#include "caf/cuda/device.hpp"
#include "caf/cuda/program.hpp"
#include <iostream>
#include <cuda.h>

namespace caf::cuda {

int device::max_active_blocks_per_sm(const program_ptr& prog,
                                     const nd_range& range,
                                     size_t dynamic_smem_bytes) const {

//	std::cout << "Hello???\n";
   try {
        if (!context_) {
            std::cerr << "[ERROR] Device context is null for device id " << id_ << "\n";
            return 0;
        }

        CUfunction kernel = prog->get_kernel(id_);
        if (!kernel) {
            std::cerr << "[ERROR] Kernel handle is null for id: " << id_ << "\n";
            return 0;
        }

        int block_size = static_cast<int>(range.get_num_threads());
        if (block_size <= 0) {
            std::cerr << "[ERROR] Block size <= 0. Block dims: "
                      << range.getBlockDimX() << "x"
                      << range.getBlockDimY() << "x"
                      << range.getBlockDimZ() << "\n";
            return 0;
        }

        // Push the device context for this thread
        CUresult res = cuCtxPushCurrent(context_);
        if (res != CUDA_SUCCESS) {
            const char* errStr = nullptr;
            cuGetErrorString(res, &errStr);
            std::cerr << "[ERROR] cuCtxPushCurrent failed: "
                      << res << " (" << (errStr ? errStr : "Unknown error") << ")\n";
            return 0;
        }

        int active_blocks = 0;
        res = cuOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks,
                                                          kernel,
                                                          block_size,
                                                          dynamic_smem_bytes);

        if (res != CUDA_SUCCESS) {
            const char* errStr = nullptr;
            cuGetErrorString(res, &errStr);
            std::cerr << "[ERROR] cuOccupancyMaxActiveBlocksPerMultiprocessor failed: "
                      << res << " (" << (errStr ? errStr : "Unknown error") << ")\n";
            active_blocks = 0;
        }

        CUcontext popped_ctx = nullptr;
        cuCtxPopCurrent(&popped_ctx);

        return active_blocks;

    } catch (const std::exception& e) {
        std::cerr << "[EXCEPTION] std::exception caught: " << e.what() << "\n";
        return 0;
    } catch (...) {
        std::cerr << "[EXCEPTION] Unknown exception caught while computing occupancy.\n";
        return 0;
    }

}
// Returns the currently available memory on this device in bytes
std::size_t device::available_memory_bytes() const {
    CUcontext prev_ctx = nullptr;
    CHECK_CUDA(cuCtxPushCurrent(context_));

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CUresult res = cuMemGetInfo(&free_bytes, &total_bytes);

    CHECK_CUDA(cuCtxPopCurrent(&prev_ctx));

    if (res != CUDA_SUCCESS) {
        const char* err_name = nullptr;
        cuGetErrorName(res, &err_name);
        throw std::runtime_error(std::string("cuMemGetInfo failed: ") + (err_name ? err_name : "unknown error"));
    }

    return free_bytes;
}

// Convenience: returns available memory in megabytes
double device::available_memory_mb() const {
    return static_cast<double>(available_memory_bytes()) / (1024.0 * 1024.0);
}

} // namespace caf::cuda

