#include <cuda.h> // CUDA Driver API
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cstdlib>
#include <thread>
#include <algorithm>
#include <functional>
#include <stdexcept> // For runtime_error
#include <unordered_map> // For MatrixPool
#include <deque>
#include <unordered_set> // For create_matrix_pool_random

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    CUresult _res = (expr);                                                    \
    if (_res != CUDA_SUCCESS) {                                                \
      const char* _err_str;                                                    \
      cuGetErrorString(_res, &_err_str);                                       \
      std::cerr << "CUDA Error in " << #expr << " at line " << __LINE__        \
                << ": " << _err_str << std::endl;                              \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

enum TaskType { MMUL = 0, VADD = 1, CONV = 2 };

struct Task {
    int N;
    TaskType type;
};

struct MatrixPool {
    std::unordered_map<int, std::vector<int>> A;
    std::unordered_map<int, std::vector<int>> B;
    std::unordered_map<int, std::vector<int>> vec_A;
    std::unordered_map<int, std::vector<int>> vec_B;
    std::unordered_map<int, std::vector<int>> conv_A;
    std::unordered_map<int, std::vector<int>> conv_K;
};

// Per-GPU execution logic
void gpu_worker(int device_id, const std::vector<Task>& tasks, int streams_per_gpu, CUcontext ctx, CUfunction mmul_func,
     CUfunction vadd_func, CUfunction conv_func, const MatrixPool& pool, int* shared_dtoh_buffer) {
    CUDA_CHECK(cuCtxSetCurrent(ctx));

    // Prepare Streams
    std::vector<CUstream> streams(streams_per_gpu);
    for (int i = 0; i < streams_per_gpu; ++i) {
        CUDA_CHECK(cuStreamCreate(&streams[i], CU_STREAM_NON_BLOCKING));
    }

    // Use the first stream for initial allocations and cleanup
    CUstream default_stream = streams[0];

    // Process assigned tasks
    for (size_t i = 0; i < tasks.size(); ++i) {
        int N = tasks[i].N;
        CUstream stream = streams[i % streams_per_gpu];
        TaskType type = tasks[i].type;
        size_t bytes_a = (size_t)N * N * sizeof(int);
        size_t bytes_b = bytes_a;
        size_t bytes_out = (size_t)N * N * sizeof(int);

        CUdeviceptr d_a, d_b, d_c;

        CUDA_CHECK(cuMemAllocAsync(&d_a, bytes_a, stream));
        CUDA_CHECK(cuMemAllocAsync(&d_b, bytes_b, stream));
        CUDA_CHECK(cuMemAllocAsync(&d_c, bytes_out, stream));

        // Perform Host-to-Device transfer
        const std::vector<int>& h_a = pool.A.at(N);
        const std::vector<int>& h_b = pool.B.at(N);

        CUDA_CHECK(cuMemcpyHtoDAsync(d_a, h_a.data(), bytes_a, stream));
        CUDA_CHECK(cuMemcpyHtoDAsync(d_b, h_b.data(), bytes_b, stream));

        // Kernel arguments for cuLaunchKernel
        void *kernel_args[] = { &d_a, &d_b, &d_c, &N };

        unsigned int block_dim = 32;
        unsigned int grid_dim = (N + block_dim - 1) / block_dim;
        CUDA_CHECK(cuLaunchKernel(mmul_func, grid_dim, grid_dim, 1,
                                      block_dim, block_dim, 1,
                                      0, stream, kernel_args, nullptr));
      


    auto h_c = std::make_shared<std::vector<int>>(N * N);
    CUDA_CHECK(cuMemcpyDtoHAsync(h_c->data(), d_c, bytes_out, stream));

    CUDA_CHECK(cuMemFreeAsync(d_a, stream));
    CUDA_CHECK(cuMemFreeAsync(d_b, stream));
    CUDA_CHECK(cuMemFreeAsync(d_c, stream));
    }

    // Synchronize this GPU context
    CUDA_CHECK(cuCtxSynchronize());

    // Cleanup
    for (auto s : streams) {
        CUDA_CHECK(cuStreamDestroy(s));
    }
}

MatrixPool create_matrix_pool_random(
    int num_sizes,
    int min_N,
    int max_N,
    unsigned int seed
) {
    MatrixPool pool;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist_N(min_N / 32, max_N / 32);
    std::unordered_set<int> used_Ns;
    while (used_Ns.size() < static_cast<size_t>(num_sizes)) {
        int N_val = dist_N(rng) * 32;
        if (N_val == 0) continue;
        if (used_Ns.insert(N_val).second) {
            pool.A[N_val] = std::vector<int>(N_val * N_val, 1);
            pool.B[N_val] = std::vector<int>(N_val * N_val, 1);
        }
    }
    return pool;
}

int main() {
    // Initialize the CUDA Driver API
    CUDA_CHECK(cuInit(0));

    const int streams_per_gpu = 8;

    int num_gpus;
    CUDA_CHECK(cuDeviceGetCount(&num_gpus));
    if (num_gpus == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return 1;
    }

    // Define parameters for irregular workload
    const int num_matrix_sizes = 10; // Number of distinct N values
    const int min_N_val = 32;
    const int max_N_val = 2048;
    const unsigned int pool_seed = 42; // Fixed seed for deterministic pool generation

    // Create the host-side matrix pool once
    MatrixPool global_host_matrix_pool = create_matrix_pool_random(num_matrix_sizes, min_N_val, max_N_val, pool_seed);

    // Extract available N values from the pool for task generation
    std::vector<int> available_Ns;
    for (const auto& pair : global_host_matrix_pool.A) {
        available_Ns.push_back(pair.first);
    }
    std::sort(available_Ns.begin(), available_Ns.end());
    if (available_Ns.empty()) {
        std::cerr << "Error: No matrix sizes generated in the pool." << std::endl;
        return 1;
    }

    std::cout << "=====================================" << std::endl;
    std::cout << "Dynamic Workload Generation (CUDA Baseline)" << std::endl;

    std::vector<CUcontext> contexts(num_gpus);
    std::vector<CUfunction> mmul_funcs(num_gpus);
    std::vector<CUfunction> vadd_funcs(num_gpus);
    std::vector<CUfunction> conv_funcs(num_gpus);
    CUmodule mmul_mod;
    CUmodule vadd_mod;
    CUmodule conv_mod;

    // Create a CUDA context for each device
    for (int i = 0; i < num_gpus; ++i) {
        CUdevice dev;
        CUDA_CHECK(cuDeviceGet(&dev, i));
        CUDA_CHECK(cuCtxCreate(&contexts[i], 0, dev));
    }

    // // Make the context for device 0 current on the main thread before loading the module
    // CUDA_CHECK(cuCtxPushCurrent(contexts[0]));

    // // Load the cubin module (assuming mmul.cu is compiled to mmul.cubin)
    // CUDA_CHECK(cuModuleLoad(&mmul_mod, "../mmul.cubin"));
    // // Get function handles
    // // CUDA_CHECK(cuModuleGetFunction(&mmul_funcs[0], mmul_mod, "matrixMul"));

    // // Pop the context from the main thread
    // CUDA_CHECK(cuCtxPopCurrent(nullptr));

    for (int i = 0; i < num_gpus; ++i) {
        CUDA_CHECK(cuCtxPushCurrent(contexts[i]));
        CUDA_CHECK(cuModuleLoad(&mmul_mod, "../mmul.cubin"));
        CUDA_CHECK(cuModuleGetFunction(&mmul_funcs[i], mmul_mod, "matrixMul"));
        CUDA_CHECK(cuCtxPopCurrent(nullptr));
    }

    std::vector<int> shared_dtoh_buffer((size_t)max_N_val * max_N_val);

    std::vector<int> batch_configs = {1};
    for (int num_batches : batch_configs) {
        std::cout << "=====================================" << std::endl;
        std::cout << "Starting Run with " << num_batches << " batches" << std::endl;

        std::mt19937 rng_prod(42); 
        std::uniform_int_distribution<size_t> dist_N_idx(0, available_Ns.size() - 1);
        std::uniform_int_distribution<int> dist_type(0, 2);
        std::uniform_int_distribution<int> dist_batch_size(50000, 50000);
        std::uniform_int_distribution<int> dist_sleep(500, 2000);

        std::vector<std::thread> all_threads;
        std::deque<std::vector<std::vector<Task>>> run_partitions_storage;

        auto start = std::chrono::steady_clock::now();

        for (int b = 0; b < num_batches; ++b) {
            // Sleep timer (random time)
            int sleep_ms = dist_sleep(rng_prod);
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));

            // Generate random sized partition (batch)
            int current_batch_size = dist_batch_size(rng_prod);
            std::vector<Task> batch_tasks;
            for (int i = 0; i < current_batch_size; ++i) {
                int N_for_task = available_Ns[dist_N_idx(rng_prod)];
                TaskType t_type = static_cast<TaskType>(0);
                batch_tasks.push_back({N_for_task, t_type});
            }

            std::cout << "Producer: Dispatching Batch " << b + 1 << "/" << num_batches 
                      << " with " << current_batch_size << " tasks..." << std::endl;

            // Static Round-Robin Partitioning for this batch
            std::vector<std::vector<Task>> partitions(num_gpus);
            for (int i = 0; i < current_batch_size; ++i) {
                partitions[i % num_gpus].push_back(batch_tasks[i]);
            }

            run_partitions_storage.push_back(std::move(partitions));
            auto& saved_partitions = run_partitions_storage.back();

            for (int i = 0; i < num_gpus; ++i) {
                all_threads.emplace_back(gpu_worker, i, std::ref(saved_partitions[i]), streams_per_gpu,
                                            contexts[i], mmul_funcs[i], vadd_funcs[i], conv_funcs[i],
                                         std::ref(global_host_matrix_pool), shared_dtoh_buffer.data());
            }
        }

        for (auto& t : all_threads) {
            t.join();
        }

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Total Makespan: " << elapsed.count() << "s" << std::endl;
    }

    // Cleanup contexts and module
    for (int i = 0; i < num_gpus; ++i) {
        CUDA_CHECK(cuCtxDestroy(contexts[i]));
    }
    // CUDA_CHECK(cuModuleUnload(mmul_mod));
    // CUDA_CHECK(cuModuleUnload(vadd_mod));
    // CUDA_CHECK(cuModuleUnload(conv_mod));

    return 0;
}
