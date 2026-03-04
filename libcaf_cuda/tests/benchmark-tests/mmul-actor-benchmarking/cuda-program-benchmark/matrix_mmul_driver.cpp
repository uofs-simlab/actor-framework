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
    std::cout << "Running N = " << N << " ...\n";
    size_t elements = static_cast<size_t>(N) * static_cast<size_t>(N);
    size_t bytes = elements * sizeof(int);

    // Check for overflow (extremely large N)
    if (elements / N != static_cast<size_t>(N)) {
        std::cerr << "Integer overflow for N = " << N << "\n";
        return;
    }

    // Host buffers
    std::vector<int> h_a, h_b, h_c;
    try {
        h_a.resize(elements);
        h_b.resize(elements);
        h_c.resize(elements);
    } catch (const std::bad_alloc&) {
        std::cerr << "Host allocation failed for N = " << N << " (need "
                  << bytes << " bytes per matrix)\n";
        return;
    }

    // Fill test data (simple pattern)
    for (size_t i = 0; i < elements; ++i) {
        h_a[i] = 1; // simple values to make it predictable
        h_b[i] = 1;
    }

    CUdeviceptr d_a = 0, d_b = 0, d_c = 0;
    CUresult r;

    // Try allocating device memory (may fail on small GPUs for large N).
    r = cuMemAlloc(&d_a, bytes);
    if (r != CUDA_SUCCESS) {
        std::cerr << "cuMemAlloc d_a failed for N=" << N << " ("
                  << bytes << " bytes). Skipping.\n";
        return;
    }
    checkCU(cuMemAlloc(&d_b, bytes), "cuMemAlloc d_b");
    checkCU(cuMemAlloc(&d_c, bytes), "cuMemAlloc d_c");

    // Grid / block
    const unsigned int blockX = 16;
    const unsigned int blockY = 16;
    unsigned int gridX = static_cast<unsigned int>((N + blockX - 1) / blockX);
    unsigned int gridY = static_cast<unsigned int>((N + blockY - 1) / blockY);

    // Prepare kernel parameter values as described by Driver API:
    // pointers to the argument values
    // Note: pass address-of CUdeviceptr, and address-of N (int).
    void* kernelParams[] = { &d_a, &d_b, &d_c, &N };

    // Time everything from H->D copy to D->H copy and sync
    auto t0 = std::chrono::steady_clock::now();

    checkCU(cuMemcpyHtoD(d_a, h_a.data(), bytes), "cuMemcpyHtoD d_a");
    checkCU(cuMemcpyHtoD(d_b, h_b.data(), bytes), "cuMemcpyHtoD d_b");

    // Launch kernel
    checkCU(cuLaunchKernel(kernel,
                           gridX, gridY, 1,         // grid
                           blockX, blockY, 1,       // block
                           0,                       // shared mem
                           nullptr,                 // stream
                           kernelParams,            // kernel params
                           nullptr), "cuLaunchKernel");

    // Wait for completion
    checkCU(cuCtxSynchronize(), "cuCtxSynchronize");

    // Copy result back
    checkCU(cuMemcpyDtoH(h_c.data(), d_c, bytes), "cuMemcpyDtoH d_c");

    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dur = t1 - t0;
    std::cout << "N=" << N << " time (alloc+H2D+kernel+DtoH): " << dur.count() << " s\n";

    // Quick spot-check for correctness on a few entries (since we used all-ones, result should be N)
    bool ok = true;
    if (elements > 0) {
        // sample first, middle, last
        std::vector<size_t> samples = {0, elements / 2, elements - 1};
        for (size_t s : samples) {
            if (h_c[s] != N) { ok = false; break; }
        }
    }
    std::cout << "Spot-check: " << (ok ? "PASS" : "FAIL (sample mismatch)") << "\n";

    // Free device memory
    checkCU(cuMemFree(d_a), "cuMemFree d_a");
    checkCU(cuMemFree(d_b), "cuMemFree d_b");
    checkCU(cuMemFree(d_c), "cuMemFree d_c");
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
