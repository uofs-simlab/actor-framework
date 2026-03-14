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

    size_t elements = (size_t)N * (size_t)N;
    size_t bytes = elements * sizeof(int);

    std::vector<int> h_a(elements, 1);
    std::vector<int> h_b(elements, 1);
    std::vector<int> h_c(elements);
    //int h_c[elements];

    CUdeviceptr d_a, d_b, d_c;
    CUstream stream;

    // ----------------------------------
    // Create Stream
    // ----------------------------------
    checkCU(cuStreamCreate(&stream, CU_STREAM_DEFAULT), "cuStreamCreate");

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

    checkCU(cuMemcpyHtoDAsync(d_a, h_a.data(), bytes, stream), "cuMemcpyHtoDAsync A");
    //checkCU(cuStreamSynchronize(stream), "sync A");

    auto t_h2d_a_end = clock::now();

    // ----------------------------------
    // H2D copy B
    // ----------------------------------
    auto t_h2d_b_start = clock::now();

    checkCU(cuMemcpyHtoDAsync(d_b, h_b.data(), bytes, stream), "cuMemcpyHtoDAsync B");
    //checkCU(cuStreamSynchronize(stream), "sync B");

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

    cuMemcpyDtoHAsync(h_c.data(), d_c, bytes, stream);
   // cuMemcpyDtoHAsync(h_c, d_c, bytes, stream);
    cuStreamSynchronize(stream);
    auto t_d2h_end = clock::now();

    auto t_total_end = clock::now();
    
    
    // ----------------------------------
    // Free device memory
    // ----------------------------------
    auto t_free_start = clock::now();

    checkCU(cuMemFree(d_a), "cuMemFree A");
    checkCU(cuMemFree(d_b), "cuMemFree B");
    checkCU(cuMemFree(d_c), "cuMemFree C");

    auto t_free_end = clock::now();




    // ----------------------------------
    // Destroy stream
    // ----------------------------------
    checkCU(cuStreamDestroy(stream), "cuStreamDestroy");


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
    
   //std::vector<int> sizes = {8000};	
   if (argc > 1) {
        sizes.clear();
        for (int i = 1; i < argc; ++i) sizes.push_back(std::stoi(argv[i]));
    }

    checkCU(cuInit(0), "cuInit");

    CUdevice dev;
    checkCU(cuDeviceGet(&dev, 0), "cuDeviceGet(0)");

    CUcontext ctx;
    checkCU(cuCtxCreate(&ctx, 0, dev), "cuCtxCreate");

    const std::string ptxPath = "mmul.ptx";
    std::string ptx;

    try {
        ptx = readFile(ptxPath);
    } catch (const std::exception &e) {
        std::cerr << "Failed to read PTX file '" << ptxPath << "': " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    CUmodule module;
    checkCU(cuModuleLoadDataEx(&module, ptx.c_str(), 0, nullptr, nullptr), "cuModuleLoadDataEx");

    CUfunction kernel;
    checkCU(cuModuleGetFunction(&kernel, module, "matrixMul"), "cuModuleGetFunction matrixMul");

    for (int N : sizes) {
        try {
            runMatrixMul(module, kernel, N);
        } catch (const std::exception &e) {
            std::cerr << "Exception while running N=" << N << ": " << e.what() << "\n";
        }
        std::cout << "----------------------------------------\n";
    }

    checkCU(cuModuleUnload(module), "cuModuleUnload");
    checkCU(cuCtxDestroy(ctx), "cuCtxDestroy");

    return 0;
}
