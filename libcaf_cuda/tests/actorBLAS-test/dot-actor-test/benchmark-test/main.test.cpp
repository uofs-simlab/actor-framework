#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <cublas_v2.h>
#include "caf/actorBLAS/dot-actor/dot-actor.hpp"

using namespace caf;
using namespace caf::cuda;

/**
 * Benchmark test for dot_actor.hpp versus native cuBLAS.
 * Measures the time taken to perform sequences of (H2D -> sdot -> D2H) operations.
 */

// Native cuBLAS benchmark implementation using Driver API and cuBLAS
void run_native_benchmark(int n, const std::vector<int>& iterations_series) {
    std::cout << "\n[INFO] Running Native cuBLAS Benchmark..." << std::endl;
    
    check(cuInit(0), "cuInit");
    CUdevice dev;
    check(cuDeviceGet(&dev, 0), "cuDeviceGet");
    CUcontext ctx;
    check(cuCtxCreate(&ctx, 0, dev), "cuCtxCreate");
    
    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "[ERROR] cublasCreate failed" << std::endl;
        return;
    }

    std::vector<float> h_x(n, 1.0f);
    std::vector<float> h_y(n, 1.0f);
    float h_res = 0.0f;
    size_t bytes = n * sizeof(float);

    for (int iters : iterations_series) {
        auto start = std::chrono::steady_clock::now();
        
        for (int i = 0; i < iters; ++i) {
            CUdeviceptr d_x, d_y, d_res;
            check(cuMemAlloc(&d_x, bytes), "cuMemAlloc d_x");
            check(cuMemAlloc(&d_y, bytes), "cuMemAlloc d_y");
            check(cuMemAlloc(&d_res, sizeof(float)), "cuMemAlloc d_res");

            check(cuMemcpyHtoD(d_x, h_x.data(), bytes), "cuMemcpyHtoD d_x");
            check(cuMemcpyHtoD(d_y, h_y.data(), bytes), "cuMemcpyHtoD d_y");

            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
            if (cublasSdot(handle, n, (const float*)d_x, 1, (const float*)d_y, 1, (float*)d_res) != CUBLAS_STATUS_SUCCESS) {
                std::cerr << "[ERROR] cublasSdot failed" << std::endl;
            }
            
            check(cuMemcpyDtoH(&h_res, d_res, sizeof(float)), "cuMemcpyDtoH d_res");

            check(cuMemFree(d_x), "cuMemFree d_x");
            check(cuMemFree(d_y), "cuMemFree d_y");
            check(cuMemFree(d_res), "cuMemFree d_res");
        }
        
        auto end = std::chrono::steady_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "[NATIVE] Iterations: " << iters << " | Time: " << diff << " ms" << std::endl;
    }

    cublasDestroy(handle);
    cuCtxDestroy(ctx);
}

// dot_actor benchmark implementation
void run_actor_benchmark(actor_system& sys, int n, const std::vector<int>& iterations_series) {
    std::cout << "\n[INFO] Running Dot Actor Benchmark..." << std::endl;
    
    scoped_actor self{sys};
    // Spawn dot_actor with reply_id = 0
    auto dot = sys.spawn<dot_actor>(0);
    
    std::vector<float> h_x(n, 1.0f);
    std::vector<float> h_y(n, 1.0f);

    for (int iters : iterations_series) {
        auto start = std::chrono::steady_clock::now();
        
        for (int i = 0; i < iters; ++i) {
            self->mail(create_in_arg(h_x), 
                       create_in_arg(h_y), 
                       create_out_arg_with_size<float>(1), 
                       n).send(dot);
            
            self->receive(
                [&](int rid, float result) {
                    // Result received via message
                }
            );
        }
        
        auto end = std::chrono::steady_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "[ACTOR]  Iterations: " << iters << " | Time: " << diff << " ms" << std::endl;
    }
    
    self->send_exit(dot, exit_reason::user_shutdown);
}

void caf_main(actor_system& sys) {
    // Initialize CUDA manager with cuBLAS support
    manager::init(sys, manager_config(true));

    int n = 1000000; // 1M elements
    std::vector<int> series = {1, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000};

    run_native_benchmark(n, series);
    run_actor_benchmark(sys, n, series);

    manager::shutdown();
}

CAF_MAIN(id_block::cuda)
