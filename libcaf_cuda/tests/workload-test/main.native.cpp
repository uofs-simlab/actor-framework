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
#include <filesystem>
#include <memory>
#include <algorithm>
#include <condition_variable>
#include <atomic>

#include "sparse_utils.hpp"

namespace fs = std::filesystem;

// ------------------------------------------------------------
// Error checking macros
// ------------------------------------------------------------
#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t status = call;                                         \
        if (status != cudaSuccess) {                                       \
            std::cerr << "CUDA Error: " << cudaGetErrorString(status)      \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

#define CHECK_CUBLAS(call)                                                 \
    do {                                                                   \
        cublasStatus_t status = call;                                      \
        if (status != CUBLAS_STATUS_SUCCESS) {                             \
            std::cerr << "cuBLAS Error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

#define CHECK_CUSPARSE(call)                                               \
    do {                                                                   \
        cusparseStatus_t status = call;                                    \
        if (status != CUSPARSE_STATUS_SUCCESS) {                           \
            std::cerr << "cuSPARSE Error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

// ------------------------------------------------------------
// Thread-safe queue
// ------------------------------------------------------------
template <typename T>
class ThreadSafeQueue {
public:
    void push(T item) {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (closed_) {
                return;
            }
            q_.push(std::move(item));
        }
        cv_.notify_one();
    }

    bool wait_pop(T& item) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [&] { return closed_ || !q_.empty(); });

        if (q_.empty()) {
            return false; // closed and empty
        }

        item = std::move(q_.front());
        q_.pop();
        return true;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            closed_ = true;
        }
        cv_.notify_all();
    }

private:
    std::mutex mtx_;
    std::condition_variable cv_;
    std::queue<T> q_;
    bool closed_ = false;
};

// ------------------------------------------------------------
// Stream slot owned by one GPU consumer thread
// ------------------------------------------------------------
struct StreamSlot {
    cudaStream_t stream{};
    cublasHandle_t cublas{};
    cusparseHandle_t cusparse{};
    cudaEvent_t done{};
    bool busy = false;
};

// ------------------------------------------------------------
// Your existing solver functions can stay the same
// ------------------------------------------------------------
// Keep these as you already have them, or minimally adjust if needed.
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



// ------------------------------------------------------------
// GPU consumer thread: one thread per GPU
// ------------------------------------------------------------
void gpu_consumer(int device_id, int num_streams, ThreadSafeQueue<MatrixTask>& work_queue) {
    CHECK_CUDA(cudaSetDevice(device_id));

    std::vector<StreamSlot> slots(num_streams);

    for (int i = 0; i < num_streams; ++i) {
        CHECK_CUDA(cudaStreamCreateWithFlags(&slots[i].stream, cudaStreamNonBlocking));
        CHECK_CUBLAS(cublasCreate(&slots[i].cublas));
        CHECK_CUSPARSE(cusparseCreate(&slots[i].cusparse));
        CHECK_CUDA(cudaEventCreateWithFlags(&slots[i].done, cudaEventDisableTiming));
        slots[i].busy = false;
    }

    int rr = 0;
    auto acquire_slot = [&]() -> int {
        while (true) {
            for (int offset = 0; offset < num_streams; ++offset) {
                int idx = (rr + offset) % num_streams;

                if (!slots[idx].busy) {
                    rr = (idx + 1) % num_streams;
                    return idx;
                }

                cudaError_t q = cudaEventQuery(slots[idx].done);
                if (q == cudaSuccess) {
                    slots[idx].busy = false;
                    rr = (idx + 1) % num_streams;
                    return idx;
                } else if (q != cudaErrorNotReady) {
                    std::cerr << "CUDA event query failed on GPU " << device_id
                              << ", stream " << idx << ": " << cudaGetErrorString(q) << std::endl;
                    std::exit(1);
                }
            }
            std::this_thread::yield();
        }
    };

    MatrixTask task;
    while (work_queue.wait_pop(task)) {
        int slot_id = acquire_slot();
        auto& slot = slots[slot_id];

        auto start_task = std::chrono::steady_clock::now();

        if (task.type == CGS_SOLVER) {
            solve_cg_async(slot.cublas, slot.cusparse, task, slot.stream);
        } 
        else {
            throw std::runtime_error("Unsupported solver type");
            exit(1);
        }

        CHECK_CUDA(cudaEventRecord(slot.done, slot.stream));
        slot.busy = true;

        auto end_task = std::chrono::steady_clock::now();
        std::chrono::duration<double> task_duration = end_task - start_task;

        std::string solver_type_str = (task.type == CGS_SOLVER) ? "CGS_SOLVER" : "BICGSTAB_SOLVER";
        std::cout << "GPU " << device_id
                  << " stream " << slot_id
                  << ": " << task.path
                  << " (" << solver_type_str << ") took "
                  << task_duration.count() << " s\n";
    }

    for (auto& slot : slots) {
        if (slot.busy) {
            CHECK_CUDA(cudaEventSynchronize(slot.done));
        }
        CHECK_CUDA(cudaEventDestroy(slot.done));
        CHECK_CUBLAS(cublasDestroy(slot.cublas));
        CHECK_CUSPARSE(cusparseDestroy(slot.cusparse));
        CHECK_CUDA(cudaStreamDestroy(slot.stream));
    }
}

// ------------------------------------------------------------
// Producer thread: enqueue pre-generated tasks
// ------------------------------------------------------------
void producer(ThreadSafeQueue<MatrixTask>& work_queue, std::vector<MatrixTask> tasks) {
    for (auto& t : tasks) {
        work_queue.push(std::move(t));
    }
    work_queue.close();
}

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------
int main(int argc, char** argv) {
    int num_streams = 8;
    if (argc > 1) {
        num_streams = std::max(1, std::atoi(argv[1]));
    }

    std::cout << "[INFO] Loading matrices...\n";
    std::vector<MatrixTask> tasks =
        scan_for_matrices("/scratch/nqr159/matrix-collection/matrix_corpus_v2/matrices/spd",
                          CGS_SOLVER);

    if (tasks.empty()) {
        std::cerr << "No matrices found.\n";
        return 1;
    }

    int num_gpus = 0;
    CHECK_CUDA(cudaGetDeviceCount(&num_gpus));
    if (num_gpus <= 0) {
        std::cerr << "No CUDA devices found.\n";
        return 1;
    }

    std::cout << "[INFO] Tasks ready: " << tasks.size() << "\n";
    std::cout << "[INFO] GPUs: " << num_gpus << "\n";
    std::cout << "[INFO] Streams per GPU: " << num_streams << "\n";

    ThreadSafeQueue<MatrixTask> work_queue;

    // Timing starts after matrix scanning/generation is already done.
    auto start = std::chrono::steady_clock::now();

    std::thread prod_thread(producer, std::ref(work_queue), std::move(tasks));

    std::vector<std::thread> consumers;
    consumers.reserve(num_gpus);
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        consumers.emplace_back(gpu_consumer, gpu, num_streams, std::ref(work_queue));
    }

    prod_thread.join();
    for (auto& t : consumers) {
        t.join();
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "\n===== NATIVE BENCHMARK COMPLETE =====\n";
    std::cout << "Tasks Processed: " << tasks.size() << "\n";
    std::cout << "Total Runtime:   " << elapsed.count() << " s\n";
    std::cout << "======================================\n";

    return 0;
}