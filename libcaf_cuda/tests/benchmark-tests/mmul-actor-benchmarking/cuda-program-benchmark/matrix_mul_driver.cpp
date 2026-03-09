// matrix_mul_driver.cpp
#include <cuda.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>

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

void runMatrixMul(CUmodule module, CUfunction kernel, int N) {
    using clock = std::chrono::steady_clock;
    using ms = std::chrono::duration<double, std::milli>;

    std::cout << "\n===== DRIVER BENCHMARK (N=" << N << ") =====\n";

    size_t elements = static_cast<size_t>(N) * N;
    size_t bytes = elements * sizeof(int);

    std::vector<int> h_a(elements, 1);
    std::vector<int> h_b(elements, 1);
    std::vector<int> h_c(elements);

    CUdeviceptr d_a, d_b, d_c;

    auto t_total_start = clock::now();

    // ----------------------------------
    // Device Allocation
    // ----------------------------------
    auto t_alloc_start = clock::now();

    checkCU(cuMemAlloc(&d_a, bytes), "cuMemAlloc d_a");
    checkCU(cuMemAlloc(&d_b, bytes), "cuMemAlloc d_b");
    checkCU(cuMemAlloc(&d_c, bytes), "cuMemAlloc d_c");

    auto t_alloc_end = clock::now();

    // ----------------------------------
    // H2D copy A
    // ----------------------------------
    auto t_h2d_a_start = clock::now();
    checkCU(cuMemcpyHtoDAsync(d_a, h_a.data(), bytes), "cuMemcpyHtoD A");
    auto t_h2d_a_end = clock::now();

    // ----------------------------------
    // H2D copy B
    // ----------------------------------
    auto t_h2d_b_start = clock::now();
    checkCU(cuMemcpyHtoDAsync(d_b, h_b.data(), bytes), "cuMemcpyHtoD B");
    auto t_h2d_b_end = clock::now();

    // ----------------------------------
    // Kernel launch + execution
    // ----------------------------------
    const unsigned int blockX = 16;
    const unsigned int blockY = 16;
    unsigned int gridX = (N + blockX - 1) / blockX;
    unsigned int gridY = (N + blockY - 1) / blockY;

    void* kernelParams[] = { &d_a, &d_b, &d_c, &N };

    auto t_kernel_start = clock::now();

    checkCU(cuLaunchKernel(kernel,
                           gridX, gridY, 1,
                           blockX, blockY, 1,
                           0,
                           nullptr,
                           kernelParams,
                           nullptr),
            "cuLaunchKernel");

    checkCU(cuCtxSynchronize(), "cuCtxSynchronize");

    auto t_kernel_end = clock::now();

    // ----------------------------------
    // D2H copy
    // ----------------------------------
    auto t_d2h_start = clock::now();
    checkCU(cuMemcpyDtoH(h_c.data(), d_c, bytes), "cuMemcpyDtoH");
    auto t_d2h_end = clock::now();

    // ----------------------------------
    // Free device memory
    // ----------------------------------
    auto t_free_start = clock::now();
    checkCU(cuMemFree(d_a), "cuMemFree A");
    checkCU(cuMemFree(d_b), "cuMemFree B");
    checkCU(cuMemFree(d_c), "cuMemFree C");
    auto t_free_end = clock::now();

    auto t_total_end = clock::now();

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
    // optional: accept single N from command line
    std::vector<int> sizes = {1000, 4000, 8000, 12000};
    if (argc > 1) {
        sizes.clear();
        for (int i = 1; i < argc; ++i) sizes.push_back(std::stoi(argv[i]));
    }

    // Initialize driver
    checkCU(cuInit(0), "cuInit");

    // pick device 0
    CUdevice dev;
    checkCU(cuDeviceGet(&dev, 0), "cuDeviceGet(0)");
    CUcontext ctx;
    checkCU(cuCtxCreate(&ctx, 0, dev), "cuCtxCreate");

    // find the PTX file next to the executable or in current dir - we assume "mmul.ptx" is present
    const std::string ptxPath = "mmul.ptx";
    std::string ptx;
    try {
        ptx = readFile(ptxPath);
    } catch (const std::exception &e) {
        std::cerr << "Failed to read PTX file '" << ptxPath << "': " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    // load module
    CUmodule module;
    checkCU(cuModuleLoadDataEx(&module, ptx.c_str(), 0, nullptr, nullptr), "cuModuleLoadDataEx");

    // get function handle
    CUfunction kernel;
    checkCU(cuModuleGetFunction(&kernel, module, "matrixMul"), "cuModuleGetFunction matrixMul");

    // Run for each size
    for (int N : sizes) {
        try {
            runMatrixMul(module, kernel, N);
        } catch (const std::exception &e) {
            std::cerr << "Exception while running N=" << N << ": " << e.what() << "\n";
        }
        std::cout << "----------------------------------------\n";
    }

    // cleanup
    checkCU(cuModuleUnload(module), "cuModuleUnload");
    checkCU(cuCtxDestroy(ctx), "cuCtxDestroy");
    return 0;
}
