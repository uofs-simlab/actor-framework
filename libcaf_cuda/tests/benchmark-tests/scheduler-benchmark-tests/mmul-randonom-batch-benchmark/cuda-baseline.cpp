#include <cuda.h> // CUDA Driver API
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <thread>
#include <algorithm>
#include <stdexcept> // For runtime_error

struct Task {
    int N;
};

// Per-GPU execution logic
void gpu_worker(int device_id, const std::vector<Task>& tasks, int streams_per_gpu, 
                CUcontext ctx, CUfunction kernel_func, const std::vector<int>& h_256, const std::vector<int>& h_2048) {
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
        const int* h_src = (N == 256) ? h_256.data() : h_2048.data();
        cuMemcpyHtoDAsync(d_a, h_src, bytes, stream);
        cuMemcpyHtoDAsync(d_b, h_src, bytes, stream);

        // Kernel arguments for cuLaunchKernel
        void *kernel_args[] = { &d_a, &d_b, &d_c, &N };

        // Kernel dimensions
        dim3 threads(32, 32);
        dim3 blocks((N + 31) / 32, (N + 31) / 32);

        cuLaunchKernel(kernel_func, blocks.x, blocks.y, blocks.z,
                       threads.x, threads.y, threads.z,
                       0, stream, kernel_args, nullptr);
        
        // Simulating the result retrieval (Copy back)
        std::vector<int> h_res(N * N);
        cuMemcpyDtoHAsync(reinterpret_cast<CUdeviceptr>(h_res.data()), d_c, bytes, stream);

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

    int num_gpus;
    cuDeviceGetCount(&num_gpus);
    
    if (num_gpus == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return 1;
    }

    const int total_tasks = 30000;
    const int streams_per_gpu = 8;

    std::vector<Task> all_tasks;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::cout << "Generating " << total_tasks << " tasks (10% Heavy, 90% Light)..." << std::endl;
    for (int i = 0; i < total_tasks; ++i) {
        if (dist(rng) < 0.1) all_tasks.push_back({2048});
        else all_tasks.push_back({256});
    }

    // Prepare Host-side MatrixPool (on CPU)
    std::vector<int> h_256(256 * 256, 1);
    std::vector<int> h_2048(2048 * 2048, 1);

    std::vector<CUcontext> contexts(num_gpus);
    std::vector<CUfunction> kernel_funcs(num_gpus);
    CUmodule module;

    // Load the cubin module (assuming mmul.cu is compiled to mmul.cubin)
    err = cuModuleLoad(&module, "../mmul.cubin");
    if (err != CUDA_SUCCESS) {
        const char* err_str;
        cuGetErrorString(err, &err_str);
        std::cerr << "Error loading module ../mmul.cubin: " << err_str << std::endl;
        return 1;
    }

    // Get function handle for matrixMul
    err = cuModuleGetFunction(&kernel_funcs[0], module, "matrixMul");
    if (err != CUDA_SUCCESS) {
        const char* err_str;
        cuGetErrorString(err, &err_str);
        std::cerr << "Error getting function matrixMul: " << err_str << std::endl;
        return 1;
    }
    // Assuming all GPUs can use the same function handle from the same module.
    for (int i = 1; i < num_gpus; ++i) {
        kernel_funcs[i] = kernel_funcs[0];
    }

    // Create a CUDA context for each device
    for (int i = 0; i < num_gpus; ++i) {
        CUdevice dev;
        cuDeviceGet(&dev, i);
        cuCtxCreate(&contexts[i], 0, dev); // Flag 0 for default context creation
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Static Round-Robin Partitioning
    // ─────────────────────────────────────────────────────────────────────────
    std::vector<std::vector<Task>> partitions(num_gpus);
    for (int i = 0; i < total_tasks; ++i) {
        partitions[i % num_gpus].push_back(all_tasks[i]);
    }

    std::cout << "Starting Static Partitioning Benchmark on " << num_gpus << " GPUs..." << std::endl;
    auto start = std::chrono::steady_clock::now();

    std::vector<std::thread> threads;
    for (int i = 0; i < num_gpus; ++i) { // Pass context and kernel function to each worker
        threads.emplace_back(gpu_worker, i, std::ref(partitions[i]), streams_per_gpu, 
                            contexts[i], kernel_funcs[i], std::ref(h_256), std::ref(h_2048));
    }

    for (auto& t : threads) {
        t.join();
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Static CUDA Complete." << std::endl;
    std::cout << "Makespan: " << elapsed.count() << "s" << std::endl;

    // Cleanup contexts and module
    for (int i = 0; i < num_gpus; ++i) {
        cuCtxDestroy(contexts[i]);
    }
    cuModuleUnload(module);

    return 0;
}
