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

#include <iostream>
#include <string>
#include <stdexcept>

// Clean Runtime API Check Macro
#define CHECK_CUDA_THROW(call)                                                 \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::string error_msg =                                            \
                std::string("[CUDA RUNTIME ERROR] API Call '") + #call +       \
                "' failed.\n  Error Name:        " +                           \
                cudaGetErrorName(err) +                                        \
                "\n  Error Description: " + cudaGetErrorString(err) +          \
                "\n  Location:          " + __FILE__ + ":" +                   \
                std::to_string(__LINE__) + "\n";                               \
            std::cerr << error_msg << std::endl;                               \
            throw std::runtime_error(error_msg);                               \
        }                                                                      \
    } while (0)

// Clean Driver API Check Macro
#define CHECK_CUDA_DRV_THROW(call)                                             \
    do {                                                                       \
        CUresult res = call;                                                   \
        if (res != CUDA_SUCCESS) {                                             \
            const char* errorName = nullptr;                                   \
            const char* errorStr = nullptr;                                    \
            cuGetErrorName(res, &errorName);                                   \
            cuGetErrorString(res, &errorStr);                                  \
            std::string error_msg =                                            \
                std::string("[CUDA DRIVER ERROR] API Call '") + #call +        \
                "' failed.\n  Error Code:        " +                           \
                std::to_string(static_cast<int>(res)) +                        \
                "\n  Error Name:        " +                                    \
                (errorName ? errorName : "UNKNOWN") +                          \
                "\n  Error Description: " +                                    \
                (errorStr ? errorStr : "UNKNOWN") +                            \
                "\n  Location:          " + __FILE__ + ":" +                   \
                std::to_string(__LINE__) + "\n";                               \
            std::cerr << error_msg << std::endl;                               \
            throw std::runtime_error(error_msg);                               \
        }                                                                      \
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
#include <vector>

// Global container to hold onto the allocated pressure pointers so they stick around
// and don't get cleaned up until the program completely finishes.
static std::vector<void*> global_pressure_allocations;
static std::mutex pressure_mutex;

void apply_memory_pressure(int device_id, size_t target_free_bytes) {
    // 1. Explicitly switch the current host thread context to the target GPU
    CHECK_CUDA_THROW(cudaSetDevice(device_id));

    size_t free_mem = 0;
    size_t total_mem = 0;
    
    // 2. Query free/total memory specifically for the newly active device
    CHECK_CUDA_THROW(cudaMemGetInfo(&free_mem, &total_mem));

    if (free_mem > target_free_bytes) {
        size_t to_allocate = free_mem - target_free_bytes;
        void* d_pressure = nullptr;
        
        // 3. Allocate the pressure block on the selected device's VRAM
        if (cudaMalloc(&d_pressure, to_allocate) == cudaSuccess) {
            std::cout << "[MAIN] Device " << device_id << " memory pressure applied. Allocated " 
                      << to_allocate / (1024 * 1024) << " MB. Roughly "
                      << target_free_bytes / (1024 * 1024) << " MB left free.\n";
            
            // Track the allocation pointer so it stays locked in memory
            std::lock_guard<std::mutex> lock(pressure_mutex);
            global_pressure_allocations.push_back(d_pressure);
        } else {
            std::cerr << "[WARNING] Failed to apply memory pressure on Device " << device_id << ".\n";
        }
    } else {
        std::cout << "[MAIN] Device " << device_id << " already has less free memory ("
                  << free_mem / (1024 * 1024) << " MB) than target ("
                  << target_free_bytes / (1024 * 1024) << " MB). Skipping.\n";
    }
}


// Worker thread logic
void worker_thread_fun(int thread_id, int device_id, TaskQueue& queue, const MatrixPool& pool, Stats& stats, CUmodule module) {
    // 1. In the Driver API, we must explicitly fetch or activate the primary context for this device
    CUdevice device;
    CUcontext context;
    if (cuDeviceGet(&device, device_id) != CUDA_SUCCESS || 
        cuDevicePrimaryCtxRetain(&context, device) != CUDA_SUCCESS) {
        std::cerr << "[THREAD " << thread_id << "] Failed to retain primary context.\n";
        return;
    }

    // Bind this thread to the device's context
    if (cuCtxSetCurrent(context) != CUDA_SUCCESS) {
        std::cerr << "[THREAD " << thread_id << "] Failed to set context.\n";
        cuDevicePrimaryCtxRelease(device);
        return;
    }

    // 2. Fetch kernel function handle
    CUfunction matrixMulKernel;
    if (cuModuleGetFunction(&matrixMulKernel, module, "matrixMul") != CUDA_SUCCESS) {
        std::cerr << "[THREAD " << thread_id << "] Failed to find 'matrixMul' kernel symbol.\n";
        cuDevicePrimaryCtxRelease(device);
        return;
    }

    // 3. Create an asynchronous stream using Driver API (CUstream)
    CUstream stream;
    if (cuStreamCreate(&stream, CU_STREAM_DEFAULT) != CUDA_SUCCESS) {
        std::cerr << "[THREAD " << thread_id << "] Failed to create stream.\n";
        cuDevicePrimaryCtxRelease(device);
        return;
    }

    int N = 0;
    while (queue.pop(N)) {
        // Driver API pointers are defined as CUdeviceptr (which is an unsigned long long / uintptr_t)
        CUdeviceptr d_A = 0;
        CUdeviceptr d_B = 0;
        CUdeviceptr d_C = 0;
        size_t size_bytes = N * N * sizeof(int);

        try {
            // 4. Stream-ordered allocations using Driver API
            CHECK_CUDA_DRV_THROW(cuMemAllocAsync(&d_A, size_bytes, stream));
            CHECK_CUDA_DRV_THROW(cuMemAllocAsync(&d_B, size_bytes, stream));
            CHECK_CUDA_DRV_THROW(cuMemAllocAsync(&d_C, size_bytes, stream));

            // 5. Asynchronous host-to-device memory copies
            CHECK_CUDA_DRV_THROW(cuMemcpyHtoDAsync(d_A, pool.A.at(N).data(), size_bytes, stream));
            CHECK_CUDA_DRV_THROW(cuMemcpyHtoDAsync(d_B, pool.B.at(N).data(), size_bytes, stream));

            // 6. Launch Kernel
            unsigned int gridX = (N + 31) / 32;
            unsigned int gridY = (N + 31) / 32;

            // Pack arguments (passing pointers to the CUdeviceptr handles)
            void* args[] = { &d_A, &d_B, &d_C, &N };

            CHECK_CUDA_DRV_THROW(
                cuLaunchKernel(matrixMulKernel,
                               gridX, gridY, 1,       // Grid dims
                               32, 32, 1,             // Block dims
                               0,                     // Shared memory bytes
                               stream,                // Stream handle
                               args,                  // Arguments
                               nullptr)               // Extra parameters
            );

            // 7. Synchronize the stream via Driver API to catch any execution anomalies
            CHECK_CUDA_DRV_THROW(cuStreamSynchronize(stream));
            stats.succeeded++;

        } catch (const std::exception& e) {
            // Because we used pure Driver API, an exception caught here does NOT poison 
            // the whole global runtime state. The thread can safely loop and request the next task.
            stats.failed++;
        }

        // 8. Stream-ordered deallocations using Driver API
        if (d_A) cuMemFreeAsync(d_A, stream);
        if (d_B) cuMemFreeAsync(d_B, stream);
        if (d_C) cuMemFreeAsync(d_C, stream);
        
        // // Ensure frees are complete before this specific worker thread loops back
        // cuStreamSynchronize(stream);
    }

    // Clean up local thread environment variables cleanly
    cuStreamDestroy(stream);
    cuDevicePrimaryCtxRelease(device);
}






int main() {
    // 1. Initialize CUDA Driver API
    if (cuInit(0) != CUDA_SUCCESS) {
        std::cerr << "Failed to initialize CUDA Driver API.\n";
        return -1;
    }

    // 2. Discover total available physical GPUs
    int num_devices = 0;
    CHECK_CUDA_THROW(cudaGetDeviceCount(&num_devices));
    if (num_devices == 0) {
        std::cerr << "CRITICAL: No CUDA-capable devices found.\n";
        return -1;
    }

    const int min_N = 2048;
    const int max_N = 4096;
    const int num_distinct_sizes = 10;
    const int num_tasks = 3000;
    
    const int STREAMS_PER_DEVICE = 32;
    const int total_worker_threads = num_devices * STREAMS_PER_DEVICE;

    // 3. Prepare task datasets
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

    // Allocate an isolated module handle tracker array for each device
    std::vector<CUmodule> modules(num_devices);

    // 4. Localize configurations per device card completely
    for (int d = 0; d < num_devices; ++d) {
        // Apply Virtual VRAM pressure boundaries to device 'd'
        apply_memory_pressure(d, 1500ULL * 1024ULL * 1024ULL); 

        // CRITICAL FIX: Explicitly enter context 'd' to load its local code copy
        CHECK_CUDA_THROW(cudaSetDevice(d));
        if (cuModuleLoad(&modules[d], "../mmul.cubin") != CUDA_SUCCESS) {
            std::cerr << "CRITICAL: Failed to load cubin module on Device " << d << "\n";
            return -1;
        }
    }

    Stats stats;
    std::cout << "\n=========================================================\n";
    std::cout << "Starting benchmark execution with " << num_tasks << " matrix jobs\n";
    std::cout << "  Devices found:       " << num_devices << "\n";
    std::cout << "  Streams per device:  " << STREAMS_PER_DEVICE << "\n";
    std::cout << "  Total async workers: " << total_worker_threads << "\n";
    std::cout << "=========================================================\n\n";

    auto start_time = std::chrono::steady_clock::now();

    // 5. Spawn worker threads passing the matching targeted hardware context module copy
    std::vector<std::thread> workers;
    workers.reserve(total_worker_threads);

    for (int d = 0; d < num_devices; ++d) {
        for (int s = 0; s < STREAMS_PER_DEVICE; ++s) {
            int global_thread_id = (d * STREAMS_PER_DEVICE) + s;

            workers.emplace_back(
                worker_thread_fun, 
                global_thread_id, 
                d, // Target device binding
                std::ref(queue), 
                std::ref(pool), 
                std::ref(stats), 
                modules[d] // Pass the distinct module loaded for this specific GPU context
            );
        }
    }

    // 6. Await execution complete
    for (auto& worker : workers) {
        if (worker.joinable()) worker.join();
    }

    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // 7. Output Final Metrics Reports
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

    // 8. Resource Deallocation Sweep
    {
        std::lock_guard<std::mutex> lock(pressure_mutex);
        for (size_t i = 0; i < global_pressure_allocations.size(); ++i) {
            if (global_pressure_allocations[i]) {
                int assigned_device = static_cast<int>(i); 
                if (assigned_device < num_devices) {
                    cudaSetDevice(assigned_device);
                }
                cudaFree(global_pressure_allocations[i]);
            }
        }
        global_pressure_allocations.clear();
    }

    // Unload each distinct module handle in its corresponding context
    for (int d = 0; d < num_devices; ++d) {
        cudaSetDevice(d);
        cuModuleUnload(modules[d]);
    }

    return 0;
}
