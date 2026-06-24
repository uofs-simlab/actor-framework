#include <cuda.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <stdexcept>

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

// Launch kernel once
void launchKernel(CUfunction kernel, CUstream stream,
                  CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_c, int N) {
    const unsigned int blockX = 32;
    const unsigned int blockY = 32;
    unsigned int gridX = (N + blockX - 1) / blockX;
    unsigned int gridY = (N + blockY - 1) / blockY;

    void* kernelParams[] = { &d_a, &d_b, &d_c, &N };

    checkCU(cuLaunchKernel(kernel,
                           gridX, gridY, 1,
                           blockX, blockY, 1,
                           0,
                           stream,
                           kernelParams,
                           nullptr),
            "cuLaunchKernel");
}

int main() {
    const int N = 1000;
    std::vector<int> iteration_series = {1000, 2000, 3000, 4000, 5000,
                                         6000, 7000, 8000, 9000, 10000};

    checkCU(cuInit(0), "cuInit");

    CUdevice dev;
    checkCU(cuDeviceGet(&dev, 0), "cuDeviceGet(0)");

    CUcontext ctx;
  
#if CUDA_VERSION >= 13000
    {
      CUctxCreateParams ctx_params = {};
      checkCU(cuCtxCreate(&ctx, &ctx_params, CU_CTX_SCHED_AUTO | CU_CTX_MAP_HOST, dev),"create context");
    }
#else
      checkCU(cuCtxCreate(&ctx, CU_CTX_SCHED_AUTO | CU_CTX_MAP_HOST, dev),"create context");
#endif

    std::string ptx = readFile("../mmul.cubin");

    CUmodule module;
    checkCU(cuModuleLoadData(&module, ptx.c_str()), "cuModuleLoadData");

    CUfunction kernel;
    checkCU(cuModuleGetFunction(&kernel, module, "matrixMul"), "cuModuleGetFunction matrixMul");

    // ----------------------------------
    // Persistent host buffers
    // ----------------------------------
    size_t elements = (size_t)N * N;
    std::vector<int> h_a(elements, 1);
    std::vector<int> h_b(elements, 1);
    std::vector<int> h_c(elements, 0);

    CUstream stream;
    checkCU(cuStreamCreate(&stream, CU_STREAM_DEFAULT), "cuStreamCreate");

    using clock = std::chrono::steady_clock;

    for (int iterations : iteration_series) {
        auto start = clock::now();

        for (int i = 0; i < iterations; ++i) {
            // ----------------------------------
            // Allocate device memory each iteration
            // ----------------------------------
            CUdeviceptr d_a, d_b, d_c;
            checkCU(cuMemAllocAsync(&d_a, elements * sizeof(int), stream), "cuMemAllocAsync d_a");
            checkCU(cuMemAllocAsync(&d_b, elements * sizeof(int), stream), "cuMemAllocAsync d_b");
            checkCU(cuMemAllocAsync(&d_c, elements * sizeof(int), stream), "cuMemAllocAsync d_c");

            // ----------------------------------
            // Copy persistent host buffers to device
            // ----------------------------------
            checkCU(cuMemcpyHtoDAsync(d_a, h_a.data(), elements * sizeof(int), stream), "H2D d_a");
            checkCU(cuMemcpyHtoDAsync(d_b, h_b.data(), elements * sizeof(int), stream), "H2D d_b");

            // ----------------------------------
            // Launch kernel
            // ----------------------------------
            launchKernel(kernel, stream, d_a, d_b, d_c, N);

            // ----------------------------------
            // Copy result back
            // ----------------------------------
            checkCU(cuMemcpyDtoHAsync(h_c.data(), d_c, elements * sizeof(int), stream), "D2H d_c");

            // ----------------------------------
            // Free device memory
            // ----------------------------------
            checkCU(cuMemFreeAsync(d_a, stream), "cuMemFreeAsync d_a");
            checkCU(cuMemFreeAsync(d_b, stream), "cuMemFreeAsync d_b");
            checkCU(cuMemFreeAsync(d_c, stream), "cuMemFreeAsync d_c");
        }

        // Synchronize stream after series
        checkCU(cuStreamSynchronize(stream), "stream sync after series");

        auto end = clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "[SERIES RESULT] Matrix " << N << "x" << N
                  << ", iterations = " << iterations
                  << ", total GPU time = " << total_ms << " ms\n";
    }

    // ----------------------------------
    // Cleanup
    // ----------------------------------
    checkCU(cuStreamDestroy(stream), "cuStreamDestroy");
    checkCU(cuModuleUnload(module), "cuModuleUnload");
    checkCU(cuCtxDestroy(ctx), "cuCtxDestroy");

    return 0;
}
