// matrix_mul_driver.cpp
#include <cuda.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <atomic>
#include <thread>

struct TimingState {
    std::chrono::steady_clock::time_point end_time;
    std::atomic<bool> ready{false};
};

void completion_callback(void* userData) {
    auto* state = static_cast<TimingState*>(userData);
    state->end_time = std::chrono::steady_clock::now();
    state->ready = true;
}

static void checkCU(CUresult r, const char* where) {
    if (r != CUDA_SUCCESS) {
        const char *str = nullptr;
        cuGetErrorString(r, &str);
        std::cerr << "CUDA Driver API error at " << where << " -> "
                  << (str ? str : "unknown") << " (" << (int)r << ")\n";
        std::exit(EXIT_FAILURE);
    }
}

std::string readFile(const std::string &path) {
    std::ifstream in(path, std::ios::in | std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::vector<int> h_c; // Declared globally

void runMatrixMul(CUmodule module, CUfunction kernel, int N, CUstream stream) {
    using clock = std::chrono::steady_clock;
    using ms = std::chrono::duration<double, std::milli>;

    std::cout << "\n===== DRIVER BENCHMARK (N=" << N << ") =====\n";

    size_t elements = (size_t)N * (size_t)N;
    size_t bytes = elements * sizeof(int);

    std::vector<int> h_a(elements, 1); // These remain local as they are initialized with N
    std::vector<int> h_b(elements, 1); // These remain local as they are initialized with N
    h_c.resize(elements); // Resize the global h_c for the current N

    CUdeviceptr d_a, d_b, d_c;

    auto t_total_start = clock::now();

    // ----------------------------------
    // Device Allocation
    // ----------------------------------
    auto t_alloc_start = clock::now();

    checkCU(cuMemAllocAsync(&d_a, bytes, stream), "cuMemAllocAsync d_a");
    checkCU(cuMemAllocAsync(&d_b, bytes, stream), "cuMemAllocAsync d_b");
    checkCU(cuMemAllocAsync(&d_c, bytes, stream), "cuMemAllocAsync d_c");

    auto t_alloc_end = clock::now();

    // ----------------------------------
    // H2D copy A
    // ----------------------------------
    auto t_h2d_a_start = clock::now();

    checkCU(cuMemcpyHtoDAsync(d_a, h_a.data(), bytes, stream), "cuMemcpyHtoDAsync A");
    std::cout << "  (Transfer size: " << bytes << " bytes)\n";

    auto t_h2d_a_end = clock::now();

    // ----------------------------------
    // H2D copy B
    // ----------------------------------
    auto t_h2d_b_start = clock::now();

    checkCU(cuMemcpyHtoDAsync(d_b, h_b.data(), bytes, stream), "cuMemcpyHtoDAsync B");
    //checkCU(cuStreamSynchronize(stream), "sync B");
    std::cout << "  (Transfer size: " << bytes << " bytes)\n";

    auto t_h2d_b_end = clock::now();

    // ----------------------------------
    // Kernel launch + execution
    // ----------------------------------
    const unsigned int blockX = 32;
    const unsigned int blockY = 32;
    unsigned int gridX = (N + blockX - 1) / blockX;
    unsigned int gridY = (N + blockY - 1) / blockY;

    void* kernelParams[] = { &d_a, &d_b, &d_c, &N };

    auto t_kernel_start = clock::now();

    checkCU(cuLaunchKernel(kernel,
                           gridX, gridY, 1,
                           blockX, blockY, 1,
                           0,
                           stream,
                           kernelParams,
                           nullptr),
            "cuLaunchKernel");

   // checkCU(cuStreamSynchronize(stream), "kernel sync");

    auto t_kernel_end = clock::now();

    // ----------------------------------
    // D2H copy
    // ----------------------------------
    auto t_d2h_start = clock::now();
    TimingState t_state;

    cuMemcpyDtoHAsync(h_c.data(), d_c, bytes, stream);

    // Enqueue the host function to capture timing when the copy finishes
    checkCU(cuLaunchHostFunc(stream, completion_callback, &t_state), "cuLaunchHostFunc");

    // In a real actor, we would not wait here. 
    // For this benchmark driver, we wait for the callback to fire.
    while (!t_state.ready) {
        std::this_thread::yield();
    }

    std::cout << "  (Transfer size: " << bytes << " bytes)\n";
    
    auto t_d2h_end = t_state.end_time;
    auto t_total_end = t_state.end_time;

    
    // ----------------------------------
    // Free device memory
    // ----------------------------------
    auto t_free_start = clock::now();

    checkCU(cuMemFreeAsync(d_a, stream), "cuMemFreeAsync A");
    checkCU(cuMemFreeAsync(d_b, stream), "cuMemFreeAsync B");
    checkCU(cuMemFreeAsync(d_c, stream), "cuMemFreeAsync C");

    auto t_free_end = clock::now();




    // ----------------------------------
    // Print Results
    // ----------------------------------

    std::cout << "Device allocation: "
              << ms(t_alloc_end - t_alloc_start).count()
              << " ms\n";

    std::cout << "H2D copy A: "
              << ms(t_h2d_a_end - t_h2d_a_start).count()
              << " ms\n";

    std::cout << "H2D copy B: "
              << ms(t_h2d_b_end - t_h2d_b_start).count()
              << " ms\n";

    std::cout << "Kernel execution: "
              << ms(t_kernel_end - t_kernel_start).count()
              << " ms\n";

    std::cout << "D2H copy: "
              << ms(t_d2h_end - t_d2h_start).count()
              << " ms\n";

    std::cout << "Device free: "
              << ms(t_free_end - t_free_start).count()
              << " ms\n";

    std::cout << "TOTAL: "
              << ms(t_total_end - t_total_start).count()
              << " ms\n";

    std::cout << "=============================================\n";
}

int main(int argc, char** argv) {
  std::vector<int> sizes = {1000, 4000, 8000, 12000};
    
//   std::vector<int> sizes = {12000};	
   if (argc > 1) {
        sizes.clear();
        for (int i = 1; i < argc; ++i) sizes.push_back(std::stoi(argv[i]));
    }

    checkCU(cuInit(0), "cuInit");

    CUdevice dev;
    checkCU(cuDeviceGet(&dev, 0), "cuDeviceGet(0)");

    CUcontext ctx;
    checkCU(cuCtxCreate(&ctx, CU_CTX_SCHED_AUTO | CU_CTX_MAP_HOST, dev), "cuCtxCreate");

    const std::string cubinPath = "mmul.cubin";
    std::string cubin;

    try {
        cubin = readFile(cubinPath);
    } catch (const std::exception &e) {
        std::cerr << "Failed to read CUBIN file '" << cubinPath << "': " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    CUmodule module;
    checkCU(cuModuleLoadDataEx(&module, cubin.data(), 0, nullptr, nullptr), "cuModuleLoadDataEx");

    CUfunction kernel;
    checkCU(cuModuleGetFunction(&kernel, module, "matrixMul"), "cuModuleGetFunction matrixMul");

    CUstream stream;
    checkCU(cuStreamCreate(&stream, CU_STREAM_DEFAULT), "cuStreamCreate");

    for (int N : sizes) {
        try {
            runMatrixMul(module, kernel, N, stream);
            
            // Ensure stream is completely empty before starting the next size
            checkCU(cuStreamSynchronize(stream), "cuStreamSynchronize between sizes");
            
        } catch (const std::exception &e) {
            std::cerr << "Exception while running N=" << N << ": " << e.what() << "\n";
        }
        std::cout << "----------------------------------------\n";
    }

    checkCU(cuStreamDestroy(stream), "cuStreamDestroy");
    checkCU(cuModuleUnload(module), "cuModuleUnload");
    checkCU(cuCtxDestroy(ctx), "cuCtxDestroy");

    return 0;
}
