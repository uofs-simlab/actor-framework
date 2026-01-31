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

} // namespace caf::cuda

