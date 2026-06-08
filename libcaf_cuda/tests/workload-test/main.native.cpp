#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <filesystem>
#include <memory>
#include <algorithm>
#include <random>
#include <atomic>
#include <cmath>

#include "sparse_utils.hpp"

namespace fs = std::filesystem;

constexpr uint32_t WORKLOAD_SEED = 42;

// ============================================================
// Error Checking
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

// ============================================================
// Existing Solver Implementations
// ============================================================

void solve_cg_async(cublasHandle_t cublas, cusparseHandle_t cusparse,
                    const MatrixTask& task, cudaStream_t stream) {
    int n = task.data->rows;
    float alpha = 1.0f, beta = 0.0f, r0 = 0.0f, r1 = 0.0f, a = 0.0f, na = 0.0f, b = 0.0f;
    float tolerance = 1e-5f;
    int max_iters = 2000;

    float *d_val, *d_x, *d_r, *d_p, *d_Ap, *d_b;
    int *d_row_ptr, *d_col_ind;

    CHECK_CUDA(cudaMallocAsync(&d_val, task.data->nnz * sizeof(float), stream));
    CHECK_CUDA(cudaMallocAsync(&d_row_ptr, (n + 1) * sizeof(int), stream));
    CHECK_CUDA(cudaMallocAsync(&d_col_ind, task.data->nnz * sizeof(int), stream));
    CHECK_CUDA(cudaMallocAsync(&d_x, n * sizeof(float), stream));
    CHECK_CUDA(cudaMallocAsync(&d_r, n * sizeof(float), stream));
    CHECK_CUDA(cudaMallocAsync(&d_p, n * sizeof(float), stream));
    CHECK_CUDA(cudaMallocAsync(&d_Ap, n * sizeof(float), stream));
    CHECK_CUDA(cudaMallocAsync(&d_b, n * sizeof(float), stream));

    CHECK_CUDA(cudaMemcpyAsync(d_val, task.data->values.data(), task.data->nnz * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_row_ptr, task.data->row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_col_ind, task.data->col_indices.data(), task.data->nnz * sizeof(int), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_b, task.data->b.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemsetAsync(d_x, 0, n * sizeof(float), stream));

    CHECK_CUBLAS(cublasSetStream(cublas, stream));
    CHECK_CUSPARSE(cusparseSetStream(cusparse, stream));

    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecP, vecAp;

    CHECK_CUSPARSE(cusparseCreateCsr(&matA, n, n, task.data->nnz, d_row_ptr, d_col_ind, d_val,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, n, d_x, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecP, n, d_p, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecAp, n, d_Ap, CUDA_R_32F));

    size_t bufferSize = 0;
    void* d_buffer = nullptr;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, matA, vecX, &beta, vecAp,
                                           CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMallocAsync(&d_buffer, bufferSize, stream));

    CHECK_CUBLAS(cublasScopy(cublas, n, d_b, 1, d_r, 1));
    CHECK_CUBLAS(cublasScopy(cublas, n, d_r, 1, d_p, 1));
    CHECK_CUBLAS(cublasSdot(cublas, n, d_r, 1, d_r, 1, &r1));

    int k = 0;
    while (k < max_iters) {
        CHECK_CUSPARSE(cusparseSpMV(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, vecP, &beta, vecAp,
                                    CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer));

        float pAp;
        CHECK_CUBLAS(cublasSdot(cublas, n, d_p, 1, d_Ap, 1, &pAp));
        a = r1 / pAp;

        CHECK_CUBLAS(cublasSaxpy(cublas, n, &a, d_p, 1, d_x, 1));

        na = -a;
        CHECK_CUBLAS(cublasSaxpy(cublas, n, &na, d_Ap, 1, d_r, 1));

        r0 = r1;
        CHECK_CUBLAS(cublasSdot(cublas, n, d_r, 1, d_r, 1, &r1));

        if (std::sqrt(r1) < tolerance) break;

        b = r1 / r0;
        CHECK_CUBLAS(cublasSscal(cublas, n, &b, d_p, 1));
        CHECK_CUBLAS(cublasSaxpy(cublas, n, &alpha, d_r, 1, d_p, 1));
        k++;
    }

    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecP));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecAp));

    CHECK_CUDA(cudaFreeAsync(d_val, stream));
    CHECK_CUDA(cudaFreeAsync(d_row_ptr, stream));
    CHECK_CUDA(cudaFreeAsync(d_col_ind, stream));
    CHECK_CUDA(cudaFreeAsync(d_x, stream));
    CHECK_CUDA(cudaFreeAsync(d_r, stream));
    CHECK_CUDA(cudaFreeAsync(d_p, stream));
    CHECK_CUDA(cudaFreeAsync(d_Ap, stream));
    CHECK_CUDA(cudaFreeAsync(d_b, stream));
    CHECK_CUDA(cudaFreeAsync(d_buffer, stream));
}

// ============================================================
// Producer
// ============================================================

void producer(
    ThreadSafeQueue<MatrixTask>& queue,
    const std::vector<MatrixTask>& matrix_pool,
    int num_batches,
    int batch_size,
    double mean_arrival_ms)
{
    std::mt19937 rng(WORKLOAD_SEED);

    for (int batch = 0; batch < num_batches; ++batch) {

        auto sleep_time =
            generate_random_interval(
                rng,
                mean_arrival_ms);

        std::this_thread::sleep_for(
            sleep_time);

        auto tasks =
            generate_batch(
                matrix_pool,
                rng,
                batch_size);

        std::cout
            << "[PRODUCER] Dispatching batch "
            << batch + 1
            << "/"
            << num_batches
            << " ("
            << tasks.size()
            << " tasks, slept "
            << sleep_time.count()
            << " ms)"
            << std::endl;

        for (auto& task : tasks) {
            queue.push(std::move(task));
        }
    }

    queue.signal_shutdown();
}

// ============================================================
// Worker
// ============================================================

void gpu_stream_worker(
    int device_id,
    int worker_id,
    ThreadSafeQueue<MatrixTask>& queue)
{
    CHECK_CUDA(cudaSetDevice(device_id));

    cudaStream_t stream;
    cublasHandle_t cublas;
    cusparseHandle_t cusparse;

    CHECK_CUDA(
        cudaStreamCreateWithFlags(
            &stream,
            cudaStreamNonBlocking));

    CHECK_CUBLAS(
        cublasCreate(&cublas));

    CHECK_CUSPARSE(
        cusparseCreate(&cusparse));

    MatrixTask task;

    while (queue.wait_pop(task)) {

        auto start_task =
            std::chrono::steady_clock::now();

        if (task.type == CGS_SOLVER) {

            solve_cg_async(
                cublas,
                cusparse,
                task,
                stream);

        } else {

            throw std::runtime_error(
                "Unsupported solver type");
        }

        CHECK_CUDA(cudaStreamSynchronize(stream));

        auto end_task =
            std::chrono::steady_clock::now();

        std::chrono::duration<double>
            elapsed = end_task - start_task;

        std::cout
            << "Worker "
            << worker_id
            << " (GPU "
            << device_id
            << ") solved "
            << task.path
            << " in "
            << elapsed.count()
            << " s"
            << std::endl;
    }

    CHECK_CUBLAS(cublasDestroy(cublas));
    CHECK_CUSPARSE(cusparseDestroy(cusparse));
    CHECK_CUDA(cudaStreamDestroy(stream));
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv)
{
    int num_streams = 4;
    int num_batches = 25;
    int batch_size = 100;
    double mean_arrival_ms = 1000.0;

    // if (argc > 1)
    //     num_streams = std::max(1, std::atoi(argv[1]));

    // if (argc > 2)
    //     num_batches = std::max(1, std::atoi(argv[2]));

    // if (argc > 3)
    //     batch_size = std::max(1, std::atoi(argv[3]));

    // if (argc > 4)
    //     mean_arrival_ms = std::atof(argv[4]);

    std::cout << "[INFO] Loading matrices...\n";

    std::vector<MatrixTask> matrix_pool =
        scan_for_matrices(
            "/scratch/nqr159/matrix-collection/matrix_corpus_v2/matrices/spd",
            CGS_SOLVER);

    if (matrix_pool.empty()) {
        std::cerr << "No matrices found.\n";
        return 1;
    }

    int num_gpus = 0;

    CHECK_CUDA(
        cudaGetDeviceCount(&num_gpus));

    if (num_gpus == 0) {
        std::cerr << "No CUDA devices found.\n";
        return 1;
    }

    std::cout
        << "[INFO] Found "
        << num_gpus
        << " GPUs\n";

    std::cout
        << "[INFO] Matrix pool size: "
        << matrix_pool.size()
        << "\n";

    std::cout
        << "[INFO] Streams/GPU: "
        << num_streams
        << "\n";

    std::cout
        << "[INFO] Batches: "
        << num_batches
        << "\n";

    std::cout
        << "[INFO] Batch size: "
        << batch_size
        << "\n";

    std::cout
        << "[INFO] Mean arrival: "
        << mean_arrival_ms
        << " ms\n";

    ThreadSafeQueue<MatrixTask> work_queue;

    auto benchmark_start =
        std::chrono::steady_clock::now();

    std::thread producer_thread(
        producer,
        std::ref(work_queue),
        std::cref(matrix_pool),
        num_batches,
        batch_size,
        mean_arrival_ms);

    std::vector<std::thread> workers;

    workers.reserve(
        num_gpus * num_streams);

    for (int gpu = 0; gpu < num_gpus; ++gpu) {

        for (int stream = 0;
             stream < num_streams;
             ++stream)
        {
            int worker_id =
                gpu * num_streams + stream;

            workers.emplace_back(
                gpu_stream_worker,
                gpu,
                worker_id,
                std::ref(work_queue));
        }
    }

    producer_thread.join();

    for (auto& worker : workers) {
        worker.join();
    }

    auto benchmark_end =
        std::chrono::steady_clock::now();

    std::chrono::duration<double>
        total_time =
            benchmark_end - benchmark_start;

    std::cout << "\n";
    std::cout << "=====================================\n";
    std::cout << "IRREGULAR WORKLOAD BENCHMARK\n";
    std::cout << "=====================================\n";
    std::cout << "Seed:               " << WORKLOAD_SEED << "\n";
    std::cout << "GPUs:               " << num_gpus << "\n";
    std::cout << "Streams per GPU:    " << num_streams << "\n";
    std::cout << "Worker Threads:     " << num_gpus * num_streams << "\n";
    std::cout << "Batches:            " << num_batches << "\n";
    std::cout << "Batch Size:         " << batch_size << "\n";
    std::cout << "Mean Arrival (ms):  " << mean_arrival_ms << "\n";
    std::cout << "Total Runtime:      " << total_time.count() << " s\n";
    std::cout << "=====================================\n";

    return 0;
}