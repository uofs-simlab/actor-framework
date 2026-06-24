/*
 * A file full of caf cuda tests that can only be verified by
 * inspection
 */

#include <caf/all.hpp>
#include <caf/actor_system.hpp>
#include <caf/cuda/manager.hpp>
#include <caf/cuda/helpers.hpp>
#include <caf/cuda/types.hpp>
#include <caf/cuda/command_runner.hpp>
#include <caf/cuda/mem_ref.hpp>
#include <caf/cuda/device.hpp>
#include <caf/cuda/streampool.hpp>
#include <caf/cuda/program.hpp>
#include <caf/cuda/nd_range.hpp>
#include <caf/cuda/platform.hpp>
#include <cassert>
#include <vector>
#include <string>
#include <cuda.h>
#include <stdexcept>
#include <iostream>


// Structure to hold test information
struct Test {
    std::string name;
    void (*function)(caf::actor_system&);
};

// Print all device properties using the device getters we added.
void test_device_display_info([[maybe_unused]] caf::actor_system& sys) {
    using namespace caf::cuda;

    try {
        auto& mgr = manager::get();
        auto dev = mgr.find_device(0);
        if (!dev) {
            std::cerr << "[test_device_display_info] No device found with id 0 — skipping\n";
            return;
        }

        std::cout << "==== Device Summary (short) ====\n";
        std::cout << dev->device_summary() << "\n\n";

        std::cout << "==== Device Detailed Properties ====\n";
        std::cout << "Device name: " << dev->name() << "\n";
        std::cout << "Device ID: " << dev->getId() << "\n";
        std::cout << "CUdevice handle: " << dev->getDevice() << "\n";
        std::cout << "CUcontext handle: " << dev->getContext() << "\n\n";

        std::cout << "Number of SMs: " << dev->num_sms() << "\n";
        std::cout << "Warp size: " << dev->warp_size() << "\n";
        std::cout << "Max threads per SM: " << dev->max_threads_per_sm() << "\n";
        std::cout << "Warps per SM (derived): " << dev->warps_per_sm() << "\n";
        std::cout << "Total warps on device (derived): " << dev->total_warps() << "\n";
        std::cout << "Total device memory (bytes): " << dev->total_memory_bytes() << "\n";
        std::cout << "Total device memory (MB): " << dev->total_memory_mb() << "\n";

        std::cout << "\n[test_device_display_info] Finished printing device info.\n";
    } catch (const std::exception& e) {
        std::cerr << "[test_device_display_info] ERROR: " << e.what() << "\n";
    } catch (...) {
        std::cerr << "[test_device_display_info] ERROR: unknown exception\n";
    }
}


void test_print_mmul_occupancy([[maybe_unused]] caf::actor_system& sys) {
  using namespace caf::cuda;

  auto& mgr = manager::get();
  auto dev = mgr.find_device(0);

  if (!dev) {
    std::cerr << "[mmul occupancy] No CUDA device found\n";
    return;
  }

  program_ptr prog;
  try {
    prog = mgr.create_program_from_cubin("../mmul.cubin", "matrixMul");
  } catch (const std::exception& e) {
    std::cerr << "[mmul occupancy] Failed to load cubin: "
              << e.what() << "\n";
    return;
  }

  // Example problem size (does not affect occupancy directly)
  constexpr int N = 1024;
  constexpr int THREADS = 32;
  const int BLOCKS = (N + THREADS - 1) / THREADS;

  nd_range dims(
      BLOCKS, BLOCKS, 1,
      THREADS, THREADS, 1
  );

  try {
    int blocks_per_sm =
        dev->max_active_blocks_per_sm(prog, dims);

    std::cout << "\n[mmul occupancy]\n";
    std::cout << "Device ID              : " << dev->getId() << "\n";
    std::cout << "Kernel                  : matrixMul\n";
    std::cout << "Threads per block       : "
              << dims.getBlockDimX() * dims.getBlockDimY() * dims.getBlockDimZ()
              << "\n";
    std::cout << "Block dims              : ("
              << dims.getBlockDimX() << ", "
              << dims.getBlockDimY() << ", "
              << dims.getBlockDimZ() << ")\n";
    std::cout << "Grid dims               : ("
              << dims.getGridDimX() << ", "
              << dims.getGridDimY() << ", "
              << dims.getGridDimZ() << ")\n";
    std::cout << "Max active blocks / SM  : "
              << blocks_per_sm << "\n";
    std::cout << "Total active blocks     : "
              << blocks_per_sm * dev->num_sms() << "\n\n";

  } catch (const std::exception& e) {
    std::cerr << "[mmul occupancy] Error querying occupancy: "
              << e.what() << "\n";
  }
}









// Return codes: 0 = PASS, 1 = SKIPPED, 2 = FAIL
int run_test(const Test& test, caf::actor_system& sys) {
    std::cout << "Running test: " << test.name << "... ";
    try {
        test.function(sys);
        std::cout << "DONE\n";
        return 0;
    } catch (const std::exception& e) {
        std::string msg = e.what();
        if (msg.find("Skipping") != std::string::npos || msg.find("skip") != std::string::npos) {
            std::cout << "SKIPPED: " << msg << "\n";
            return 1;
        }
        std::cout << "FAILED: " << msg << "\n";
        return 2;
    } catch (...) {
        std::cout << "FAILED: Unknown error\n";
        return 2;
    }
}

// Test registry — add more tests here as needed.
const std::vector<Test> tests = {
    {"test_device_display_info", test_device_display_info},
    {"test_print_mmul_occupancy", test_print_mmul_occupancy}
};

// CAF main function to run tests
void caf_main(caf::actor_system& sys) {
    // Initialize CUDA manager
    try {
        caf::cuda::manager::init(sys);
        std::cout << "CUDA manager initialized successfully\n";
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize CUDA manager: " << e.what() << "\n";
        return;
    }

    // Run tests
    std::cout << "\nStarting unit tests...\n\n";
    int passed = 0;
    int skipped = 0;
    int failed = 0;

    for (const auto& test : tests) {
        int status = run_test(test, sys);
        if (status == 0) ++passed;
        else if (status == 1) ++skipped;
        else ++failed;
    }

    // Shutdown CUDA manager
    try {
        caf::cuda::manager::shutdown();
        std::cout << "\nCUDA manager shutdown successfully\n";
    } catch (const std::exception& e) {
        std::cerr << "Failed to shutdown CUDA manager: " << e.what() << "\n";
    }

    // Summary
    std::cout << "\nTest Summary:\n";
    std::cout << "Total tests listed: " << tests.size() << "\n";
    std::cout << "Passed: " << passed << "\n";
    std::cout << "Skipped: " << skipped << "\n";
    std::cout << "Failed: " << failed << "\n";

    if (failed > 0) {
        // Throw to indicate failure to the test runner / CI environment if present.
        throw std::runtime_error("One or more tests failed");
    }
}

// Register caf_main
CAF_MAIN()

