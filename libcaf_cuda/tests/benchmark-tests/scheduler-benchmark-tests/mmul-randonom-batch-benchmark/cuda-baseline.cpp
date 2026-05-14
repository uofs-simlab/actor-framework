#include <cuda.h> // CUDA Driver API
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <thread>
#include <algorithm>
#include <functional>
#include <stdexcept> // For runtime_error
#include <unordered_map> // For MatrixPool
#include <unordered_set> // For create_matrix_pool_random

struct Task {
    int N;
};

// Per-GPU execution logic
void gpu_worker(int device_id, const std::vector<Task>& tasks, int streams_per_gpu, 
                CUcontext ctx, CUfunction kernel_func, const std::unordered_map<int, std::vector<int>>& host_matrix_A, const std::unordered_map<int, std::vector<int>>& host_matrix_B) {
    // Set the CUDA context for this thread
    CUresult err = cuCtxSetCurrent(ctx);
    if (err != CUDA_SUCCESS) {
        const char* err_str;
        cuGetErrorString(err, &err_str);
        std::cerr << "Error setting context for device " << device_id << ": " << err_str << std::endl;
        return;
    }

    // Prepare Streams
    std::vector<CUstream> streams(streams_per_gpu);
    for (int i = 0; i < streams_per_gpu; ++i) {
        cuStreamCreate(&streams[i], CU_STREAM_NON_BLOCKING);
    }

    // Use the first stream for initial allocations and cleanup
    CUstream default_stream = streams[0];

    // Process assigned tasks
    for (size_t i = 0; i < tasks.size(); ++i) {
        int N = tasks[i].N;
        CUstream stream = streams[i % streams_per_gpu];
        size_t bytes = static_cast<size_t>(N) * N * sizeof(int);

        CUdeviceptr d_a, d_b, d_c;

        // Allocate GPU memory for this specific task
        cuMemAllocAsync(&d_a, bytes, stream);
        cuMemAllocAsync(&d_b, bytes, stream);
        cuMemAllocAsync(&d_c, bytes, stream);

        // Perform Host-to-Device transfer
        cuMemcpyHtoDAsync(d_a, host_matrix_A.at(N).data(), bytes, stream);
        cuMemcpyHtoDAsync(d_b, host_matrix_B.at(N).data(), bytes, stream);

        // Kernel arguments for cuLaunchKernel
        void *kernel_args[] = { &d_a, &d_b, &d_c, &N };

        // Kernel dimensions
        unsigned int block_dim = 32;
        unsigned int grid_dim = (N + block_dim - 1) / block_dim;

        cuLaunchKernel(kernel_func, grid_dim, grid_dim, 1,
                       block_dim, block_dim, 1,
                       0, stream, kernel_args, nullptr);
        
        // Simulating the result retrieval (Copy back)
        std::vector<int> h_res(N * N);
        cuMemcpyDtoHAsync(h_res.data(), d_c, bytes, stream);

        // Free GPU memory for this task
        cuMemFreeAsync(d_a, stream);
        cuMemFreeAsync(d_b, stream);
        cuMemFreeAsync(d_c, stream);
    }

    // Synchronize this GPU context
    cuCtxSynchronize();

    // Cleanup
    for (auto s : streams) {
        cuStreamDestroy(s);
    }
}

struct MatrixPool {
    std::unordered_map<int, std::vector<int>> A;
    std::unordered_map<int, std::vector<int>> B;
};

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
    CUresult err;

    // Initialize the CUDA Driver API
    err = cuInit(0);
    if (err != CUDA_SUCCESS) {
        const char* err_str;
        cuGetErrorString(err, &err_str);
        std::cerr << "Error initializing CUDA Driver API: " << err_str << std::endl;
        return 1;
    }

    const int streams_per_gpu = 8;
    std::vector<int> task_counts = {50000,100000};

    int num_gpus;
    cuDeviceGetCount(&num_gpus);
    if (num_gpus == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return 1;
    }

    // Define parameters for irregular workload
    const int num_matrix_sizes = 60; // Number of distinct N values
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
    if (available_Ns.empty()) {
        std::cerr << "Error: No matrix sizes generated in the pool." << std::endl;
        return 1;
    }

    for (int total_tasks : task_counts) {
    std::cout << "=====================================" << std::endl;
    std::cout << "Task count: " << total_tasks << " (Irregular Workload)" << std::endl;

    std::vector<Task> all_tasks;
    std::mt19937 rng_tasks(42); // Fixed seed for task distribution
    std::uniform_int_distribution<size_t> dist_N_idx(0, available_Ns.size() - 1);

    std::vector<CUcontext> contexts(num_gpus);
    std::vector<CUfunction> kernel_funcs(num_gpus);
    CUmodule module;

    // Create a CUDA context for each device
    for (int i = 0; i < num_gpus; ++i) {
        CUdevice dev;
        cuDeviceGet(&dev, i);
        err = cuCtxCreate(&contexts[i], 0, dev); // Flag 0 for default context creation
        if (err != CUDA_SUCCESS) {
            const char* err_str;
            cuGetErrorString(err, &err_str);
            std::cerr << "Error creating context for device " << i << ": " << err_str << std::endl;
            // Clean up already created contexts
            for (int j = 0; j < i; ++j) cuCtxDestroy(contexts[j]);
            return 1;
        }
    }

    // Make the context for device 0 current on the main thread before loading the module
    err = cuCtxPushCurrent(contexts[0]);
    if (err != CUDA_SUCCESS) {
        const char* err_str;
        cuGetErrorString(err, &err_str);
        std::cerr << "Error pushing context for device 0: " << err_str << std::endl;
        for (int i = 0; i < num_gpus; ++i) cuCtxDestroy(contexts[i]);
        return 1;
    }

    // Load the cubin module (assuming mmul.cu is compiled to mmul.cubin)
    err = cuModuleLoad(&module, "../mmul.cubin");
    if (err != CUDA_SUCCESS) {
        const char* err_str;
        cuGetErrorString(err, &err_str);
        std::cerr << "Error loading module ../mmul.cubin: " << err_str << std::endl;
        cuCtxPopCurrent(nullptr); // Pop context on error
        for (int i = 0; i < num_gpus; ++i) cuCtxDestroy(contexts[i]);
        return 1;
    }

    // Get function handle for matrixMul
    err = cuModuleGetFunction(&kernel_funcs[0], module, "matrixMul");
    if (err != CUDA_SUCCESS) {
        const char* err_str;
        cuGetErrorString(err, &err_str);
        std::cerr << "Error getting function matrixMul: " << err_str << std::endl;
        cuCtxPopCurrent(nullptr); // Pop context on error
        cuModuleUnload(module);
        for (int i = 0; i < num_gpus; ++i) cuCtxDestroy(contexts[i]);
        return 1;
    }

    // Pop the context from the main thread
    err = cuCtxPopCurrent(nullptr);
    if (err != CUDA_SUCCESS) {
        const char* err_str;
        cuGetErrorString(err, &err_str);
        std::cerr << "Error popping context from main thread: " << err_str << std::endl;
        cuModuleUnload(module);
        for (int i = 0; i < num_gpus; ++i) cuCtxDestroy(contexts[i]);
        return 1;
    }

    // Assuming all GPUs can use the same function handle from the same module.
    for (int i = 1; i < num_gpus; ++i) {
        kernel_funcs[i] = kernel_funcs[0];
    }

    for (int i = 0; i < total_tasks; ++i) {
        int N_for_task = available_Ns[dist_N_idx(rng_tasks)];
        all_tasks.push_back({N_for_task});
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Static Round-Robin Partitioning
    // ─────────────────────────────────────────────────────────────────────────
    std::vector<std::vector<Task>> partitions(num_gpus);
    for (int i = 0; i < total_tasks; ++i) {
        partitions[i % num_gpus].push_back(all_tasks[i]);
    }

    auto start = std::chrono::steady_clock::now();

    std::vector<std::thread> threads;
    for (int i = 0; i < num_gpus; ++i) { // Pass context and kernel function to each worker
        threads.emplace_back(gpu_worker, i, std::ref(partitions[i]), streams_per_gpu, contexts[i], kernel_funcs[i], std::ref(global_host_matrix_pool.A), std::ref(global_host_matrix_pool.B));
    }

    for (auto& t : threads) {
        t.join();
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Makespan: " << elapsed.count() << "s" << std::endl;

    // Cleanup contexts and module
    for (int i = 0; i < num_gpus; ++i) {
        cuCtxDestroy(contexts[i]);
    }
    cuModuleUnload(module);
    }

    return 0;
}
