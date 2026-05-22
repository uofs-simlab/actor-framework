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
        TaskType type = tasks[i].type;
        size_t bytes_a = (type == MMUL) ? (size_t)N * N * sizeof(int) : (size_t)N * sizeof(int);
        size_t bytes_b = (type == CONV) ? 5 * sizeof(int) : bytes_a;
        size_t bytes_out = (type == MMUL) ? (size_t)N * N * sizeof(int) : (size_t)N * sizeof(int);

        CUdeviceptr d_a, d_b, d_c;

        cuMemAllocAsync(&d_a, bytes_a, stream);
        cuMemAllocAsync(&d_b, bytes_b, stream);
        cuMemAllocAsync(&d_c, bytes_out, stream);

        // Perform Host-to-Device transfer
        const std::vector<int>& h_a = (type == MMUL) ? pool.A.at(N) : (type == VADD ? pool.vec_A.at(N) : pool.conv_A.at(N));
        const std::vector<int>& h_b = (type == MMUL) ? pool.B.at(N) : (type == VADD ? pool.vec_B.at(N) : pool.conv_K.at(N));

        cuMemcpyHtoDAsync(d_a, h_a.data(), bytes_a, stream);
        cuMemcpyHtoDAsync(d_b, h_b.data(), bytes_b, stream);

        // Kernel arguments for cuLaunchKernel
        void *kernel_args[] = { &d_a, &d_b, &d_c, &N };

        if (type == MMUL) {
            unsigned int block_dim = 32;
            unsigned int grid_dim = (N + block_dim - 1) / block_dim;
            cuLaunchKernel(mmul_func, grid_dim, grid_dim, 1,
                           block_dim, block_dim, 1,
                           0, stream, kernel_args, nullptr);
        } else if (type == VADD) {
            unsigned int block_dim = 256;
            unsigned int grid_dim = (N + block_dim - 1) / block_dim;
            cuLaunchKernel(vadd_func, grid_dim, 1, 1,
                           block_dim, 1, 1,
                           0, stream, kernel_args, nullptr);
        } else {
            unsigned int block_dim = 256;
            unsigned int grid_dim = (N + block_dim - 1) / block_dim;
            cuLaunchKernel(conv_func, grid_dim, 1, 1,
                           block_dim, 1, 1,
                           0, stream, kernel_args, nullptr);
        }
        
        // Simulating the result retrieval (Copy back)
        size_t res_count = (type == MMUL) ? (size_t)N * N : (size_t)N;
        cuMemcpyDtoHAsync(shared_dtoh_buffer, d_c, bytes_out, stream);

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
            pool.vec_A[N_val] = std::vector<int>(N_val, 1);
            pool.vec_B[N_val] = std::vector<int>(N_val, 1);
            pool.conv_A[N_val] = std::vector<int>(N_val, 1);
            pool.conv_K[N_val] = std::vector<int>(5, 1);
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
    std::sort(available_Ns.begin(), available_Ns.end());
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
    std::vector<CUfunction> mmul_funcs(num_gpus);
    std::vector<CUfunction> vadd_funcs(num_gpus);
    std::vector<CUfunction> conv_funcs(num_gpus);
    CUmodule mmul_mod;
    CUmodule vadd_mod;
    CUmodule conv_mod;

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
    err = cuModuleLoad(&mmul_mod, "../mmul.cubin");
    if (err != CUDA_SUCCESS) {
        const char* err_str;
        cuGetErrorString(err, &err_str);
        std::cerr << "Error loading module ../mmul.cubin: " << err_str << std::endl;
        cuCtxPopCurrent(nullptr); // Pop context on error
        return 1;
    }

    err = cuModuleLoad(&vadd_mod, "../vector_add.cubin");
    if (err != CUDA_SUCCESS) {
        const char* err_str;
        cuGetErrorString(err, &err_str);
        std::cerr << "Error loading module ../vector_add.cubin: " << err_str << std::endl;
        cuCtxPopCurrent(nullptr);
        return 1;
    }

    err = cuModuleLoad(&conv_mod, "../conv1d.cubin");
    if (err != CUDA_SUCCESS) {
        const char* err_str;
        cuGetErrorString(err, &err_str);
        std::cerr << "Error loading module ../conv1d.cubin: " << err_str << std::endl;
        cuCtxPopCurrent(nullptr);
        return 1;
    }

    // Get function handles
    err = cuModuleGetFunction(&mmul_funcs[0], mmul_mod, "matrixMul");
    if (err != CUDA_SUCCESS) {
        const char* err_str;
        cuGetErrorString(err, &err_str);
        std::cerr << "Error getting function matrixMul: " << err_str << std::endl;
        cuCtxPopCurrent(nullptr); // Pop context on error
        cuModuleUnload(mmul_mod);
        cuModuleUnload(vadd_mod);
        for (int i = 0; i < num_gpus; ++i) cuCtxDestroy(contexts[i]);
        return 1;
    }

    err = cuModuleGetFunction(&vadd_funcs[0], vadd_mod, "vectorAdd");
    if (err != CUDA_SUCCESS) {
        const char* err_str;
        cuGetErrorString(err, &err_str);
        std::cerr << "Error getting function vectorAdd: " << err_str << std::endl;
        cuCtxPopCurrent(nullptr);
        return 1;
    }

    err = cuModuleGetFunction(&conv_funcs[0], conv_mod, "conv1d");
    if (err != CUDA_SUCCESS) {
        const char* err_str;
        cuGetErrorString(err, &err_str);
        std::cerr << "Error getting function conv1d: " << err_str << std::endl;
        cuCtxPopCurrent(nullptr);
        return 1;
    }

    // Pop the context from the main thread
    err = cuCtxPopCurrent(nullptr);
    if (err != CUDA_SUCCESS) {
        const char* err_str;
        cuGetErrorString(err, &err_str);
        std::cerr << "Error popping context from main thread: " << err_str << std::endl;
        cuModuleUnload(mmul_mod);
        cuModuleUnload(vadd_mod);
        cuModuleUnload(conv_mod);
        for (int i = 0; i < num_gpus; ++i) cuCtxDestroy(contexts[i]);
        return 1;
    }

    // Assuming all GPUs can use the same function handle from the same module.
    for (int i = 1; i < num_gpus; ++i) {
        mmul_funcs[i] = mmul_funcs[0];
        vadd_funcs[i] = vadd_funcs[0];
        conv_funcs[i] = conv_funcs[0];
    }

    std::uniform_int_distribution<int> dist_type(0, 2);
    for (int i = 0; i < total_tasks; ++i) {
        int N_for_task = available_Ns[dist_N_idx(rng_tasks)];
        TaskType t_type = static_cast<TaskType>(dist_type(rng_tasks));
        all_tasks.push_back({N_for_task, t_type});
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Static Round-Robin Partitioning
    // ─────────────────────────────────────────────────────────────────────────
    std::vector<std::vector<Task>> partitions(num_gpus);
    for (int i = 0; i < total_tasks; ++i) {
        partitions[i % num_gpus].push_back(all_tasks[i]);
    }

    // Preallocate a single large host buffer for DTOH transfers to save RAM and keep things fair.
    std::vector<int> shared_dtoh_buffer((size_t)max_N_val * max_N_val);

    auto start = std::chrono::steady_clock::now();

    std::vector<std::thread> threads;
    for (int i = 0; i < num_gpus; ++i) { // Pass context and kernel function to each worker
        threads.emplace_back(gpu_worker, i, std::ref(partitions[i]), streams_per_gpu, contexts[i], mmul_funcs[i], vadd_funcs[i], conv_funcs[i],
         std::ref(global_host_matrix_pool), shared_dtoh_buffer.data());
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
    cuModuleUnload(mmul_mod);
    cuModuleUnload(vadd_mod);
    cuModuleUnload(conv_mod);
    }

    return 0;
}
