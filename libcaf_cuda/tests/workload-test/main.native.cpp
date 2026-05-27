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
#include "sparse_utils.hpp"

namespace fs = std::filesystem;

// Error checking macros
#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t status = call;                                         \
        if (status != cudaSuccess) {                                       \
            std::cerr << "CUDA Error: " << cudaGetErrorString(status)      \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

#define CHECK_CUBLAS(call)                                                 \
    do {                                                                   \
        cublasStatus_t status = call;                                      \
        if (status != CUBLAS_STATUS_SUCCESS) {                             \
            std::cerr << "cuBLAS Error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

#define CHECK_CUSPARSE(call)                                               \
    do {                                                                   \
        cusparseStatus_t status = call;                                    \
        if (status != CUSPARSE_STATUS_SUCCESS) {                           \
            std::cerr << "cuSparse Error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

enum SolverType { CGS_SOLVER, BICSTAB_SOLVER };

struct MatrixTask {
    std::string path;
    SolverType type;
    int32_t rows, cols, nnz;
    std::vector<int32_t> row_ptr;
    std::vector<int32_t> col_indices;
    std::vector<float> values;
    std::vector<float> b;
};

// Simplified CG Solver using raw cuBLAS and cuSPARSE
void solve_cg_async(cublasHandle_t cublas, cusparseHandle_t cusparse, const MatrixTask& task, cudaStream_t stream) {
    int n = task.rows;
    float alpha = 1.0f, beta = 0.0f, r0 = 0.0f, r1 = 0.0f, a = 0.0f, na = 0.0f, b = 0.0f;
    float tolerance = 1e-5f;
    int max_iters = 2000;

    float *d_val, *d_x, *d_r, *d_p, *d_Ap, *d_b;
    int *d_row_ptr, *d_col_ind;

    // Use Stream Ordered Allocator
    CHECK_CUDA(cudaMallocAsync(&d_val, task.nnz * sizeof(float), stream));
    CHECK_CUDA(cudaMallocAsync(&d_row_ptr, (n + 1) * sizeof(int), stream));
    CHECK_CUDA(cudaMallocAsync(&d_col_ind, task.nnz * sizeof(int), stream));
    CHECK_CUDA(cudaMallocAsync(&d_x, n * sizeof(float), stream));
    CHECK_CUDA(cudaMallocAsync(&d_r, n * sizeof(float), stream));
    CHECK_CUDA(cudaMallocAsync(&d_p, n * sizeof(float), stream));
    CHECK_CUDA(cudaMallocAsync(&d_Ap, n * sizeof(float), stream));
    CHECK_CUDA(cudaMallocAsync(&d_b, n * sizeof(float), stream));

    CHECK_CUDA(cudaMemcpyAsync(d_val, task.values.data(), task.nnz * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_row_ptr, task.row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_col_ind, task.col_indices.data(), task.nnz * sizeof(int), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_b, task.b.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemsetAsync(d_x, 0, n * sizeof(float), stream));

    // Create descriptors
    CHECK_CUBLAS(cublasSetStream(cublas, stream));
    CHECK_CUSPARSE(cusparseSetStream(cusparse, stream));

    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecP, vecAp;
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, n, n, task.nnz, d_row_ptr, d_col_ind, d_val, 
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, n, d_x, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecP, n, d_p, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecAp, n, d_Ap, CUDA_R_32F));

    size_t bufferSize = 0;
    void* d_buffer = nullptr;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecAp, 
                                           CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMallocAsync(&d_buffer, bufferSize, stream));

    // CG Logic: r = b - Ax (initially r = b since x=0)
    CHECK_CUBLAS(cublasScopy(cublas, n, d_b, 1, d_r, 1));
    CHECK_CUBLAS(cublasScopy(cublas, n, d_r, 1, d_p, 1));
    CHECK_CUBLAS(cublasSdot(cublas, n, d_r, 1, d_r, 1, &r1));

    int k = 0;
    while (k < max_iters) {
        // Ap = A * p
        CHECK_CUSPARSE(cusparseSpMV(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecP, &beta, vecAp, 
                                    CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer));
        
        float pAp;
        CHECK_CUBLAS(cublasSdot(cublas, n, d_p, 1, d_Ap, 1, &pAp));
        a = r1 / pAp;

        // x = x + a*p
        CHECK_CUBLAS(cublasSaxpy(cublas, n, &a, d_p, 1, d_x, 1));
        
        // r = r - a*Ap
        na = -a;
        CHECK_CUBLAS(cublasSaxpy(cublas, n, &na, d_Ap, 1, d_r, 1));

        r0 = r1;
        CHECK_CUBLAS(cublasSdot(cublas, n, d_r, 1, d_r, 1, &r1));

        if (sqrt(r1) < tolerance) break;

        b = r1 / r0;
        // p = r + b*p => scal p by b then add r
        CHECK_CUBLAS(cublasSscal(cublas, n, &b, d_p, 1));
        CHECK_CUBLAS(cublasSaxpy(cublas, n, &alpha, d_r, 1, d_p, 1));
        k++;
    }

    // Cleanup
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecP);
    cusparseDestroyDnVec(vecAp);

    // Stream-ordered free
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

// BiCGSTAB Solver using raw cuBLAS and cuSPARSE (Asynchronous version)
void solve_bicgstab_async(cublasHandle_t cublas, cusparseHandle_t cusparse, const MatrixTask& task, cudaStream_t stream) {
    int n = task.rows;
    float alpha = 1.0f, beta = 0.0f, omega = 1.0f, rho = 1.0f, rho_prev = 1.0f;
    float tolerance = 1e-5f;
    int max_iters = 2000;

    float *d_val, *d_x, *d_r, *d_r_hat, *d_p, *d_v, *d_s, *d_t, *d_b;
    int *d_row_ptr, *d_col_ind;

    CHECK_CUDA(cudaMallocAsync(&d_val, task.nnz * sizeof(float), stream));
    CHECK_CUDA(cudaMallocAsync(&d_row_ptr, (n + 1) * sizeof(int), stream));
    CHECK_CUDA(cudaMallocAsync(&d_col_ind, task.nnz * sizeof(int), stream));
    CHECK_CUDA(cudaMallocAsync(&d_x, n * sizeof(float), stream));
    CHECK_CUDA(cudaMallocAsync(&d_r, n * sizeof(float), stream));
    CHECK_CUDA(cudaMallocAsync(&d_r_hat, n * sizeof(float), stream));
    CHECK_CUDA(cudaMallocAsync(&d_p, n * sizeof(float), stream));
    CHECK_CUDA(cudaMallocAsync(&d_v, n * sizeof(float), stream));
    CHECK_CUDA(cudaMallocAsync(&d_s, n * sizeof(float), stream));
    CHECK_CUDA(cudaMallocAsync(&d_t, n * sizeof(float), stream));
    CHECK_CUDA(cudaMallocAsync(&d_b, n * sizeof(float), stream));

    CHECK_CUDA(cudaMemcpyAsync(d_val, task.values.data(), task.nnz * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_row_ptr, task.row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_col_ind, task.col_indices.data(), task.nnz * sizeof(int), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_b, task.b.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemsetAsync(d_x, 0, n * sizeof(float), stream));
    CHECK_CUDA(cudaMemsetAsync(d_v, 0, n * sizeof(float), stream));
    CHECK_CUDA(cudaMemsetAsync(d_p, 0, n * sizeof(float), stream));

    CHECK_CUBLAS(cublasSetStream(cublas, stream));
    CHECK_CUSPARSE(cusparseSetStream(cusparse, stream));

    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecP, vecV, vecS, vecT;
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, n, n, task.nnz, d_row_ptr, d_col_ind, d_val, 
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecP, n, d_p, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecV, n, d_v, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecS, n, d_s, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecT, n, d_t, CUDA_R_32F));

    size_t bufferSize = 0;
    void* d_buffer = nullptr;
    float one = 1.0f, zero = 0.0f;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, vecP, &zero, vecV, 
                                           CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMallocAsync(&d_buffer, bufferSize, stream));

    // r = b - Ax (initially r = b)
    CHECK_CUBLAS(cublasScopy(cublas, n, d_b, 1, d_r, 1));
    CHECK_CUBLAS(cublasScopy(cublas, n, d_r, 1, d_r_hat, 1));

    for (int i = 1; i <= max_iters; ++i) {
        CHECK_CUBLAS(cublasSdot(cublas, n, d_r_hat, 1, d_r, 1, &rho));
        
        if (i == 1) {
            CHECK_CUBLAS(cublasScopy(cublas, n, d_r, 1, d_p, 1));
        } else {
            beta = (rho / rho_prev) * (alpha / omega);
            float minus_omega = -omega;
            CHECK_CUBLAS(cublasSaxpy(cublas, n, &minus_omega, d_v, 1, d_p, 1));
            CHECK_CUBLAS(cublasSscal(cublas, n, &beta, d_p, 1));
            CHECK_CUBLAS(cublasSaxpy(cublas, n, &one, d_r, 1, d_p, 1));
        }

        // v = Ap
        CHECK_CUSPARSE(cusparseSpMV(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, vecP, &zero, vecV, 
                                    CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer));

        float rhat_v;
        CHECK_CUBLAS(cublasSdot(cublas, n, d_r_hat, 1, d_v, 1, &rhat_v));
        alpha = rho / rhat_v;

        // s = r - alpha*v
        CHECK_CUBLAS(cublasScopy(cublas, n, d_r, 1, d_s, 1));
        float neg_alpha = -alpha;
        CHECK_CUBLAS(cublasSaxpy(cublas, n, &neg_alpha, d_v, 1, d_s, 1));

        // t = As
        CHECK_CUSPARSE(cusparseSpMV(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, vecS, &zero, vecT, 
                                    CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer));

        float t_s, t_t;
        CHECK_CUBLAS(cublasSdot(cublas, n, d_t, 1, d_s, 1, &t_s));
        CHECK_CUBLAS(cublasSdot(cublas, n, d_t, 1, d_t, 1, &t_t));
        omega = t_s / t_t;

        // x = x + alpha*p + omega*s
        CHECK_CUBLAS(cublasSaxpy(cublas, n, &alpha, d_p, 1, d_x, 1));
        CHECK_CUBLAS(cublasSaxpy(cublas, n, &omega, d_s, 1, d_x, 1));

        // r = s - omega*t
        CHECK_CUBLAS(cublasScopy(cublas, n, d_s, 1, d_r, 1));
        float neg_omega_bc = -omega;
        CHECK_CUBLAS(cublasSaxpy(cublas, n, &neg_omega_bc, d_t, 1, d_r, 1));

        float norm_r;
        CHECK_CUBLAS(cublasSnrm2(cublas, n, d_r, 1, &norm_r));
        if (norm_r < tolerance) break;
        
        rho_prev = rho;
    }

    // Cleanup
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecP); cusparseDestroyDnVec(vecV);
    cusparseDestroyDnVec(vecS); cusparseDestroyDnVec(vecT);
    CHECK_CUDA(cudaFreeAsync(d_val, stream));
    CHECK_CUDA(cudaFreeAsync(d_row_ptr, stream));
    CHECK_CUDA(cudaFreeAsync(d_col_ind, stream));
    CHECK_CUDA(cudaFreeAsync(d_x, stream));
    CHECK_CUDA(cudaFreeAsync(d_r, stream));
    CHECK_CUDA(cudaFreeAsync(d_r_hat, stream));
    CHECK_CUDA(cudaFreeAsync(d_p, stream));
    CHECK_CUDA(cudaFreeAsync(d_v, stream));
    CHECK_CUDA(cudaFreeAsync(d_s, stream));
    CHECK_CUDA(cudaFreeAsync(d_t, stream));
    CHECK_CUDA(cudaFreeAsync(d_b, stream));
    CHECK_CUDA(cudaFreeAsync(d_buffer, stream));
}

// New GPU Dispatcher function (One thread per GPU)
void gpu_dispatcher(int device_id, std::vector<MatrixTask> assigned_tasks, int num_workers) {
    CHECK_CUDA(cudaSetDevice(device_id));
    
    std::vector<cudaStream_t> streams(num_workers);
    std::vector<cublasHandle_t> cublas_handles(num_workers);
    std::vector<cusparseHandle_t> cusparse_handles(num_workers);

    for (int i = 0; i < num_workers; ++i) {
        CHECK_CUDA(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
        CHECK_CUBLAS(cublasCreate(&cublas_handles[i]));
        CHECK_CUSPARSE(cusparseCreate(&cusparse_handles[i]));
    }

    // Deep pipelining: Loop through assigned tasks and dispatch to streams round-robin
    for (size_t i = 0; i < assigned_tasks.size(); ++i) {
        int s_idx = i % num_workers;
        const auto& t = assigned_tasks[i];
        if (t.type == CGS_SOLVER) {
            solve_cg_async(cublas_handles[s_idx], cusparse_handles[s_idx], t, streams[s_idx]);
        } else {
            solve_bicgstab_async(cublas_handles[s_idx], cusparse_handles[s_idx], t, streams[s_idx]); 
        }
    }

    // Wait for everything on this device to finish
    CHECK_CUDA(cudaDeviceSynchronize());

    for (int i = 0; i < num_workers; ++i) {
        cublasDestroy(cublas_handles[i]);
        cusparseDestroy(cusparse_handles[i]);
        cudaStreamDestroy(streams[i]);
    }
}

int main() {
    std::vector<MatrixTask> tasks;
    
    auto scan = [&](const std::string& dir, SolverType type) {
        if (!fs::exists(dir)) return;
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.path().extension() == ".bin") {
                auto coo = load_binary_coo(entry.path().string());
                auto csr = convert_coo_to_csr(coo);
                MatrixTask t;
                // Compute t.b using the valid csr object before its internal vectors are moved
                t.b = compute_rhs_spmv(csr, std::vector<float>(csr.cols, 1.0f));
                t.path = entry.path().string();
                t.type = type;
                t.rows = csr.rows;
                t.nnz = csr.nnz;
                t.row_ptr = std::move(csr.row_ptr);
                t.col_indices = std::move(csr.col_indices);
                t.values = std::move(csr.values);
                tasks.push_back(std::move(t));
            }
        }
    };

    std::cout << "[INFO] Loading matrices...\n";
    scan("/scratch/nqr159/matrix-collection/matrices/spd", CGS_SOLVER);
    scan("/scratch/nqr159/matrix-collection/matrices/unsymmetric", BICSTAB_SOLVER);

    if (tasks.empty()) {
        std::cerr << "No matrices found.\n";
        return 1;
    }

    int num_gpus;
    CHECK_CUDA(cudaGetDeviceCount(&num_gpus));
    int workers_per_gpu = 8;
    
    // Static Round-Robin Partitioning
    std::vector<std::vector<MatrixTask>> partitions(num_gpus);
    for (size_t i = 0; i < tasks.size(); ++i) {
        partitions[i % num_gpus].push_back(std::move(tasks[i]));
    }

    std::cout << "[INFO] Processing " << tasks.size() << " tasks using " 
              << num_gpus << " Dispatcher threads (" << workers_per_gpu << " streams/GPU)...\n";

    auto start = std::chrono::steady_clock::now();

    std::vector<std::thread> dispatchers;
    for (int i = 0; i < num_gpus; ++i) {
        dispatchers.emplace_back(gpu_dispatcher, i, std::move(partitions[i]), workers_per_gpu);
    }

    for (auto& d : dispatchers) d.join();

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "\n===== NATIVE BENCHMARK COMPLETE =====\n";
    std::cout << "Tasks Processed: " << tasks.size() << "\n";
    std::cout << "Total Runtime:   " << elapsed.count() << " s\n";
    std::cout << "======================================\n";

    return 0;
}