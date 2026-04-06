// matrix_mul_driver.cpp
#include <cuda.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <stdexcept>
#include <random>

static const unsigned int RANDOM_SEED = 42;

static void checkCU(CUresult r, const char* where) {
    if (r != CUDA_SUCCESS) {
        const char *str = nullptr;
        cuGetErrorString(r, &str);
        std::cerr << "CUDA Driver API error at " << where << " -> "
                  << (str ? str : "unknown") << " (" << (int)r << ")\n";
        std::exit(EXIT_FAILURE);
    }
}

// runMatrixMul: executes the kernel and returns the total duration in milliseconds
double runMatrixMul(CUmodule module, CUfunction kernel, int N) {
    using clock = std::chrono::steady_clock;
    using ms = std::chrono::duration<double, std::milli>;

    std::cout << "\n===== DRIVER BENCHMARK (N=" << N << ") =====\n";

    size_t elements = (size_t)N * (size_t)N;
    size_t bytes = elements * sizeof(int);

    std::mt19937 rng(RANDOM_SEED);
    std::uniform_int_distribution<int> dist(1, 10);

    std::vector<int> h_a(elements);
    std::vector<int> h_b(elements);
    std::vector<int> h_c(elements);

    for (auto& v : h_a) v = dist(rng);
    for (auto& v : h_b) v = dist(rng);

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
    checkCU(cuMemAlloc(&d_a, bytes), "cuMemAlloc d_a");
    checkCU(cuMemAlloc(&d_b, bytes), "cuMemAlloc d_b");
    checkCU(cuMemAlloc(&d_c, bytes), "cuMemAlloc d_c");

    // ----------------------------------
    // H2D copies (async, pipelined with kernel)
    // ----------------------------------
    checkCU(cuMemcpyHtoDAsync(d_a, h_a.data(), bytes, stream), "cuMemcpyHtoDAsync A");
    checkCU(cuMemcpyHtoDAsync(d_b, h_b.data(), bytes, stream), "cuMemcpyHtoDAsync B");

    // ----------------------------------
    // Kernel launch
    // ----------------------------------
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

    // ----------------------------------
    // D2H copy + synchronize all pending GPU work
    // ----------------------------------
    cuMemcpyDtoHAsync(h_c.data(), d_c, bytes, stream);
    cuStreamSynchronize(stream);

    // ----------------------------------
    // Free device memory
    // ----------------------------------
    checkCU(cuMemFree(d_a), "cuMemFree A");
    checkCU(cuMemFree(d_b), "cuMemFree B");
    checkCU(cuMemFree(d_c), "cuMemFree C");

    auto t_total_end = clock::now();

    // ----------------------------------
    // Destroy stream
    // ----------------------------------
    checkCU(cuStreamDestroy(stream), "cuStreamDestroy");

    // ----------------------------------
    // Print Results
    // (Sub-timings omitted: intermediate async ops have no sync point
    //  so CPU-side timestamps do not reflect actual GPU phase durations.)
    // ----------------------------------
    double total_ms = ms(t_total_end - t_total_start).count();

    std::cout << "TOTAL: " << total_ms << " ms\n";
    std::cout << "=============================================\n";

    return total_ms;
}

int main(int argc, char** argv) {
    std::vector<int> sizes = {1000, 2000, 4000, 8000, 16000};
    
   if (argc > 1) {
        sizes.clear();
        for (int i = 1; i < argc; ++i) sizes.push_back(std::stoi(argv[i]));
    }

    checkCU(cuInit(0), "cuInit");

    CUdevice dev;
    checkCU(cuDeviceGet(&dev, 0), "cuDeviceGet(0)");

    CUcontext ctx;
    checkCU(cuCtxCreate(&ctx, 0, dev), "cuCtxCreate");

    CUmodule module;
    checkCU(cuModuleLoad(&module, "mmul.cubin"), "cuModuleLoad mmul.cubin");

    CUfunction kernel;
    checkCU(cuModuleGetFunction(&kernel, module, "matrixMul"), "cuModuleGetFunction matrixMul");

    // Warmup: small run to prime CUDA lazy-init / cubin load before timed tests
    runMatrixMul(module, kernel, 64);
    std::cout << "--- warmup complete ---\n";

    std::vector<std::pair<int,double>> results;

    for (int N : sizes) {
        try {
            double t = runMatrixMul(module, kernel, N);
            results.emplace_back(N, t);
        } catch (const std::exception &e) {
            std::cerr << "Exception while running N=" << N << ": " << e.what() << "\n";
        }
        std::cout << "----------------------------------------\n";
    }

    // Print summary of results
    std::cout << "\nMatrix size : time (ms)\n";
    for (auto &p : results) {
        std::cout << p.first << " : " << p.second << " ms\n";
    }

    checkCU(cuModuleUnload(module), "cuModuleUnload");
    checkCU(cuCtxDestroy(ctx), "cuCtxDestroy");

    return 0;
}
