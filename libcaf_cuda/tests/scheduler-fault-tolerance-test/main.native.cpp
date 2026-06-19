#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <queue>
#include <cuda_runtime.h>
#include <cuda.h> // Include CUDA Driver API header

// Macro for safe CUDA Runtime error checking
#define CHECK_CUDA_THROW(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(cudaGetErrorString(err)); \
        } \
    } while (0)

// Macro for safe CUDA Driver error checking
#define CHECK_CUDA_DRV_THROW(call) \
    do { \
        CUresult res = call; \
        if (res != CUDA_SUCCESS) { \
            const char* errStr = nullptr; \
            cuGetErrorString(res, &errStr); \
            throw std::runtime_error(errStr ? errStr : "Unknown Driver Error"); \
        } \
    } while (0)

struct MatrixPool {
    std::unordered_map<int, std::vector<int>> A;
    std::unordered_map<int, std::vector<int>> B;
};

struct TaskQueue {
    std::queue<int> tasks;
    std::mutex mtx;

    void push(int N) {
        std::lock_guard<std::mutex> lock(mtx);
        tasks.push(N);
    }

    bool pop(int& N) {
        std::lock_guard<std::mutex> lock(mtx);
        if (tasks.empty()) return false;
        N = tasks.front();
        tasks.pop();
        return true;
    }
};

struct Stats {
    std::atomic<int> succeeded{0};
    std::atomic<int> failed{0};
};

MatrixPool create_matrix_pool_random(int num_sizes, int min_N, int max_N, unsigned int seed) {
    MatrixPool pool;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(min_N, max_N);
    std::unordered_set<int> used;

    while (used.size() < static_cast<size_t>(num_sizes)) {
        int N = dist(rng);
        if (used.insert(N).second) {
            pool.A[N] = std::vector<int>(N * N, 1);
            pool.B[N] = std::vector<int>(N * N, 2);
        }
    }
    return pool;
}

void apply_memory_pressure(size_t target_free_bytes) {
    size_t free_mem = 0;
    size_t total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);

    if (free_mem > target_free_bytes) {
        size_t to_allocate = free_mem - target_free_bytes;
        void* d_pressure = nullptr;
        
        if (cudaMalloc(&d_pressure, to_allocate) == cudaSuccess) {
            std::cout << "[MAIN] Memory pressure applied. Allocated " 
                      << to_allocate / (1024 * 1024) << " MB. Roughly "
                      << target_free_bytes / (1024 * 1024) << " MB left free.\n";
        } else {
            std::cerr << "[WARNING] Failed to apply initial memory pressure.\n";
        }
    }
}

// Worker thread logic
void worker_thread_fun(int thread_id, int device_id, TaskQueue& queue, const MatrixPool& pool, Stats& stats, CUmodule module) {
    cudaSetDevice(device_id);

    // Fetch kernel function handle from the preloaded module
    CUfunction matrixMulKernel;
    if (cuModuleGetFunction(&matrixMulKernel, module, "matrixMul") != CUDA_SUCCESS) {
        std::cerr << "[THREAD " << thread_id << "] Failed to find 'matrixMul' kernel symbol.\n";
        return;
    }

    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::cerr << "[THREAD " << thread_id << "] Failed to create stream.\n";
        return;
    }

    int N = 0;
    while (queue.pop(N)) {
        int* d_A = nullptr;
        int* d_B = nullptr;
        int* d_C = nullptr;
        size_t size_bytes = N * N * sizeof(int);

        try {
            // 1. Allocations
            CHECK_CUDA_THROW(cudaMalloc(&d_A, size_bytes));
            CHECK_CUDA_THROW(cudaMalloc(&d_B, size_bytes));
            CHECK_CUDA_THROW(cudaMalloc(&d_C, size_bytes));

            // 2. Upload host data
            CHECK_CUDA_THROW(cudaMemcpyAsync(d_A, pool.A.at(N).data(), size_bytes, cudaMemcpyHostToDevice, stream));
            CHECK_CUDA_THROW(cudaMemcpyAsync(d_B, pool.B.at(N).data(), size_bytes, cudaMemcpyHostToDevice, stream));

            // 3. Launch Kernel via Driver API
            unsigned int gridX = (N + 31) / 32;
            unsigned int gridY = (N + 31) / 32;

            // Pack arguments exactly as expected by the kernel parameters layout
            void* args[] = { &d_A, &d_B, &d_C, &N };

            CHECK_CUDA_DRV_THROW(
                cuLaunchKernel(matrixMulKernel,
                               gridX, gridY, 1,       // Grid dims
                               32, 32, 1,             // Block dims
                               0,                     // Shared memory bytes
                               stream,                // Casts cleanly to CUstream
                               args,                  // Kernel arguments
                               nullptr)               // Extra arguments
            );

            // 4. Synchronize stream to catch runtime/OOM issues execution pipeline errors
            CHECK_CUDA_THROW(cudaStreamSynchronize(stream));
            stats.succeeded++;

        } catch (const std::exception& e) {
            stats.failed++;
        }

        // Clean up resources for this job loop
        if (d_A) cudaFree(d_A);
        if (d_B) cudaFree(d_B);
        if (d_C) cudaFree(d_C);
    }

    cudaStreamDestroy(stream);
}

int main() {
    // Initialize CUDA Driver API
    if (cuInit(0) != CUDA_SUCCESS) {
        std::cerr << "Failed to initialize CUDA Driver API.\n";
        return -1;
    }

    const int min_N = 2048;
    const int max_N = 4096;
    const int num_distinct_sizes = 10;
    const int num_tasks = 3000;
    const int num_worker_threads = 32; 
    const int target_device = 0;

    MatrixPool pool = create_matrix_pool_random(num_distinct_sizes, min_N, max_N, 42);
    std::vector<int> available_Ns;
    for (const auto& pair : pool.A) {
        available_Ns.push_back(pair.first);
    }

    TaskQueue queue;
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> dist_N_idx(0, available_Ns.size() - 1);
    for (int i = 0; i < num_tasks; ++i) {
        queue.push(available_Ns[dist_N_idx(rng)]);
    }

    cudaSetDevice(target_device);

    // Load the CUBIN module once globally before processing worker loops
    CUmodule module;
    if (cuModuleLoad(&module, "../mmul.cubin") != CUDA_SUCCESS) {
        std::cerr << "CRITICAL: Failed to load cubin file from '../mmul.cubin'\n";
        return -1;
    }

    apply_memory_pressure(1500ULL * 1024ULL * 1024ULL); 

  // 1. Fetch total number of available CUDA devices
    int num_devices = 0;
    CHECK_CUDA_THROW(cudaGetDeviceCount(&num_devices));
    if (num_devices == 0) {
        std::cerr << "CRITICAL: No CUDA-capable devices found.\n";
        return -1;
    }

    // 2. Enforce exactly 32 streams/threads per device
    const int STREAMS_PER_DEVICE = 32;
    const int total_worker_threads = num_devices * STREAMS_PER_DEVICE;

    // 3. Apply memory pressure to ALL devices uniformly
    for (int d = 0; d < num_devices; ++d) {
        cudaSetDevice(d);
        apply_memory_pressure(1500ULL * 1024ULL * 1024ULL); // Keep ~1500 MB free per GPU
    }

    Stats stats;
    std::cout << "Starting benchmark execution with " << num_tasks << " matrix jobs\n"
              << "  Devices found:       " << num_devices << "\n"
              << "  Streams per device:  " << STREAMS_PER_DEVICE << "\n"
              << "  Total async workers: " << total_worker_threads << "\n"
              << "=========================================================\n";

    auto start_time = std::chrono::steady_clock::now();

    // 4. Spawn threads grouped by device to ensure precise distribution
    std::vector<std::thread> workers;
    workers.reserve(total_worker_threads);

    for (int d = 0; d < num_devices; ++d) {
        for (int s = 0; s < STREAMS_PER_DEVICE; ++s) {
            // Unique thread ID calculation for logging/tracking if needed
            int global_thread_id = (d * STREAMS_PER_DEVICE) + s;

            workers.emplace_back(
                worker_thread_fun, 
                global_thread_id, 
                d, // Explicit target device ID
                std::ref(queue), 
                std::ref(pool), 
                std::ref(stats), 
                module
            );
        }
    }

    // 5. Await processing completion across all GPU lanes
    for (auto& worker : workers) {
        if (worker.joinable()) worker.join();
    }

    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Output Stats Report
    int total_processed = stats.succeeded + stats.failed;
    double success_ratio = total_processed > 0 ? (100.0 * stats.succeeded / total_processed) : 0.0;

    std::cout << "\n=====================================\n";
    std::cout << "[STATS REPORT] Run Complete\n";
    std::cout << "  Total Processed: " << total_processed << "\n";
    std::cout << "  Total Succeeded: " << stats.succeeded.load() << "\n";
    std::cout << "  Total Failed:    " << stats.failed.load() << "\n";
    std::cout << "  Success Ratio:   " << success_ratio << "%\n";
    std::cout << "  Total Makespan:  " << elapsed.count() << " seconds\n";
    std::cout << "=====================================\n";

    // Unload the CUBIN before termination
    cuModuleUnload(module);
    return 0;
}