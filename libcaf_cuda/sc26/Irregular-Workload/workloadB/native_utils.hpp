#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <iostream>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include "sparse_utils.hpp"

constexpr int MAX_ITERATIONS = 16000;

// ============================================================
// Error Checking Macros
// ============================================================

#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t status = call;                                         \
        if (status != cudaSuccess) {                                       \
            std::cerr << "CUDA Error: " << cudaGetErrorString(status)      \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

#define CHECK_CUBLAS(call)                                                 \
    do {                                                                   \
        cublasStatus_t status = call;                                      \
        if (status != CUBLAS_STATUS_SUCCESS) {                             \
            std::cerr << "cuBLAS Error at "                               \
                      << __FILE__ << ":" << __LINE__ << std::endl;         \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

#define CHECK_CUSPARSE(call)                                               \
    do {                                                                   \
        cusparseStatus_t status = call;                                    \
        if (status != CUSPARSE_STATUS_SUCCESS) {                           \
            std::cerr << "cuSPARSE Error at "                             \
                      << __FILE__ << ":" << __LINE__ << std::endl;         \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

// ============================================================
// Thread Safe Queue
// ============================================================

template<typename T>
class ThreadSafeQueue {
public:
    void push(T item) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(item));
        }
        cv_.notify_one();
    }

    bool wait_pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] {
            return shutdown_ || !queue_.empty();
        });

        if (!queue_.empty()) {
            item = std::move(queue_.front());
            queue_.pop();
            return true;
        }
        return false;
    }

    void signal_shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            shutdown_ = true;
        }
        cv_.notify_all();
    }

private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool shutdown_ = false;
};

int solve_cg_async(cublasHandle_t cublas, cusparseHandle_t cusparse, const MatrixTask& task, cudaStream_t stream);
int solve_pcg_jacobi_async(cublasHandle_t cublas, cusparseHandle_t cusparse, const MatrixTask& task, cudaStream_t stream);
void gpu_stream_worker(int device_id, int worker_id, ThreadSafeQueue<MatrixTask>& queue, std::atomic<int>& succeeded, std::atomic<int>& failed);