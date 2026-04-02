// main_cuda_native.cpp
// Option A: Serialised latency — cuStreamSynchronize added after each D2H copy
// inside the loop so native completes one full GPU round-trip per iteration
// before queuing the next. This matches the actor variants' per-iteration
// blocking semantics exactly and gives a truly apples-to-apples overhead
// measurement. Expected overhead ratio: ~1.5–2x instead of ~4.5x.
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
    checkCU(cuCtxCreate(&ctx, 0, dev), "cuCtxCreate");

    std::string cubin = readFile("mmul.cubin");

    CUmodule module;
    checkCU(cuModuleLoadDataEx(&module, cubin.data(), 0, nullptr, nullptr), "cuModuleLoadDataEx");

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

    // Warmup: prime CUDA context and allocator before timed series
    {
        const int warmup_iters = 10;
        for (int i = 0; i < warmup_iters; ++i) {
            CUdeviceptr d_a, d_b, d_c;
            checkCU(cuMemAlloc(&d_a, elements * sizeof(int)), "warmup alloc d_a");
            checkCU(cuMemAlloc(&d_b, elements * sizeof(int)), "warmup alloc d_b");
            checkCU(cuMemAlloc(&d_c, elements * sizeof(int)), "warmup alloc d_c");
            checkCU(cuMemcpyHtoDAsync(d_a, h_a.data(), elements * sizeof(int), stream), "warmup H2D d_a");
            checkCU(cuMemcpyHtoDAsync(d_b, h_b.data(), elements * sizeof(int), stream), "warmup H2D d_b");
            launchKernel(kernel, stream, d_a, d_b, d_c, N);
            checkCU(cuMemcpyDtoHAsync(h_c.data(), d_c, elements * sizeof(int), stream), "warmup D2H d_c");
            checkCU(cuStreamSynchronize(stream), "warmup sync");  // serialised in warmup too
            checkCU(cuMemFree(d_a), "warmup free d_a");
            checkCU(cuMemFree(d_b), "warmup free d_b");
            checkCU(cuMemFree(d_c), "warmup free d_c");
        }
        std::cout << "--- warmup complete ---\n";
    }

    for (int iterations : iteration_series) {
        auto start = clock::now();

        for (int i = 0; i < iterations; ++i) {
            CUdeviceptr d_a, d_b, d_c;
            checkCU(cuMemAlloc(&d_a, elements * sizeof(int)), "cuMemAlloc d_a");
            checkCU(cuMemAlloc(&d_b, elements * sizeof(int)), "cuMemAlloc d_b");
            checkCU(cuMemAlloc(&d_c, elements * sizeof(int)), "cuMemAlloc d_c");

            checkCU(cuMemcpyHtoDAsync(d_a, h_a.data(), elements * sizeof(int), stream), "H2D d_a");
            checkCU(cuMemcpyHtoDAsync(d_b, h_b.data(), elements * sizeof(int), stream), "H2D d_b");

            launchKernel(kernel, stream, d_a, d_b, d_c, N);

            checkCU(cuMemcpyDtoHAsync(h_c.data(), d_c, elements * sizeof(int), stream), "D2H d_c");

            // [OPTION A] Synchronise after each iteration — GPU completes one
            // full round-trip before the next iteration begins. This matches
            // the actor variants' blocking request/receive semantics exactly.
            checkCU(cuStreamSynchronize(stream), "per-iteration sync");

            checkCU(cuMemFree(d_a), "cuMemFree d_a");
            checkCU(cuMemFree(d_b), "cuMemFree d_b");
            checkCU(cuMemFree(d_c), "cuMemFree d_c");
        }

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
