// main_cuda_native.cpp
// Option B: Throughput mode — run REPS repetitions per matrix size with GPU
// work pipelined (single cuStreamSynchronize after all reps for each N).
// This matches the fire-and-forget actor variant and shows the actor model can
// approach native throughput when results are not needed per-iteration.
#include <cuda.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <stdexcept>
#include <random>

static const unsigned int RANDOM_SEED = 42;
static const int REPS = 10;   // repetitions per matrix size

static void checkCU(CUresult r, const char* where) {
    if (r != CUDA_SUCCESS) {
        const char *str = nullptr;
        cuGetErrorString(r, &str);
        std::cerr << "CUDA Driver API error at " << where << " -> "
                  << (str ? str : "unknown") << " (" << (int)r << ")\n";
        std::exit(EXIT_FAILURE);
    }
}

// runMatrixMulThroughput: enqueues REPS iterations without intermediate syncs,
// then synchronises once at the end. Returns total wall-clock time in ms.
double runMatrixMulThroughput(CUmodule module, CUfunction kernel, int N) {
    using clock = std::chrono::steady_clock;
    using ms = std::chrono::duration<double, std::milli>;

    std::cout << "\n===== DRIVER BENCHMARK THROUGHPUT (N=" << N
              << ", reps=" << REPS << ") =====\n";

    size_t elements = (size_t)N * (size_t)N;
    size_t bytes = elements * sizeof(int);

    std::mt19937 rng(RANDOM_SEED);
    std::uniform_int_distribution<int> dist(1, 10);

    // Persistent host buffers — allocated once, reused across all reps
    std::vector<int> h_a(elements);
    std::vector<int> h_b(elements);
    std::vector<int> h_c(elements);

    for (auto& v : h_a) v = dist(rng);
    for (auto& v : h_b) v = dist(rng);

    CUstream stream;
    checkCU(cuStreamCreate(&stream, CU_STREAM_DEFAULT), "cuStreamCreate");

    const unsigned int blockX = 32;
    const unsigned int blockY = 32;
    unsigned int gridX = (N + blockX - 1) / blockX;
    unsigned int gridY = (N + blockY - 1) / blockY;

    // Pipeline: enqueue all REPS iterations without syncing between them.
    // Device memory is allocated fresh each rep (mirrors actor behaviour) but
    // all ops are queued to the same stream before the single final sync.
    auto t_start = clock::now();

    for (int r = 0; r < REPS; ++r) {
        CUdeviceptr d_a, d_b, d_c;
        checkCU(cuMemAlloc(&d_a, bytes), "cuMemAlloc d_a");
        checkCU(cuMemAlloc(&d_b, bytes), "cuMemAlloc d_b");
        checkCU(cuMemAlloc(&d_c, bytes), "cuMemAlloc d_c");

        checkCU(cuMemcpyHtoDAsync(d_a, h_a.data(), bytes, stream), "H2D A");
        checkCU(cuMemcpyHtoDAsync(d_b, h_b.data(), bytes, stream), "H2D B");

        void* kernelParams[] = { &d_a, &d_b, &d_c, &N };
        checkCU(cuLaunchKernel(kernel,
                               gridX, gridY, 1,
                               blockX, blockY, 1,
                               0, stream,
                               kernelParams, nullptr),
                "cuLaunchKernel");

        checkCU(cuMemcpyDtoHAsync(h_c.data(), d_c, bytes, stream), "D2H C");

        // cuMemFree is logically deferred in CUDA 12 — GPU continues executing
        checkCU(cuMemFree(d_a), "cuMemFree d_a");
        checkCU(cuMemFree(d_b), "cuMemFree d_b");
        checkCU(cuMemFree(d_c), "cuMemFree d_c");
    }

    // Single synchronisation after all reps — GPU was busy the whole time
    checkCU(cuStreamSynchronize(stream), "final stream sync");

    auto t_end = clock::now();
    double total_ms = ms(t_end - t_start).count();

    checkCU(cuStreamDestroy(stream), "cuStreamDestroy");

    std::cout << "Total for " << REPS << " reps: " << total_ms << " ms\n";
    std::cout << "Per-rep average: " << total_ms / REPS << " ms\n";
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
    runMatrixMulThroughput(module, kernel, 64);
    std::cout << "--- warmup complete ---\n";

    std::vector<std::pair<int,double>> results;

    for (int N : sizes) {
        try {
            double t = runMatrixMulThroughput(module, kernel, N);
            results.emplace_back(N, t);
        } catch (const std::exception &e) {
            std::cerr << "Exception while running N=" << N << ": " << e.what() << "\n";
        }
        std::cout << "----------------------------------------\n";
    }

    std::cout << "\nMatrix size : total time (ms) for " << REPS << " reps : per-rep avg (ms)\n";
    for (auto &p : results) {
        std::cout << p.first << " : " << p.second
                  << " ms : " << p.second / REPS << " ms\n";
    }

    checkCU(cuModuleUnload(module), "cuModuleUnload");
    checkCU(cuCtxDestroy(ctx), "cuCtxDestroy");

    return 0;
}
