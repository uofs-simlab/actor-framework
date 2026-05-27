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
void solve_cg(cublasHandle_t cublas, cusparseHandle_t cusparse, const MatrixTask& task) {
    int n = task.rows;
    float alpha = 1.0f, beta = 0.0f, r0 = 0.0f, r1 = 0.0f, a = 0.0f, na = 0.0f, b = 0.0f;
    float tolerance = 1e-5f;
    int max_iters = 2000;

    float *d_val, *d_x, *d_r, *d_p, *d_Ap, *d_b;
    int *d_row_ptr, *d_col_ind;

    CHECK_CUDA(cudaMalloc(&d_val, task.nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_col_ind, task.nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_x, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_r, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_p, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Ap, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_val, task.values.data(), task.nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_row_ptr, task.row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_ind, task.col_indices.data(), task.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, task.b.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_x, 0, n * sizeof(float)));

    // Create descriptors
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
    CHECK_CUDA(cudaMalloc(&d_buffer, bufferSize));

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
    cudaFree(d_val); cudaFree(d_row_ptr); cudaFree(d_col_ind);
    cudaFree(d_x); cudaFree(d_r); cudaFree(d_p); cudaFree(d_Ap); cudaFree(d_b);
    cudaFree(d_buffer);
}

// BiCGSTAB Solver using raw cuBLAS and cuSPARSE
void solve_bicgstab(cublasHandle_t cublas, cusparseHandle_t cusparse, const MatrixTask& task) {
    int n = task.rows;
    float alpha = 1.0f, beta = 0.0f, omega = 1.0f, rho = 1.0f, rho_prev = 1.0f;
    float tolerance = 1e-5f;
    int max_iters = 2000;

    float *d_val, *d_x, *d_r, *d_r_hat, *d_p, *d_v, *d_s, *d_t, *d_b;
    int *d_row_ptr, *d_col_ind;

    CHECK_CUDA(cudaMalloc(&d_val, task.nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_col_ind, task.nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_x, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_r, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_r_hat, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_p, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_v, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_s, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_t, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_val, task.values.data(), task.nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_row_ptr, task.row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_ind, task.col_indices.data(), task.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, task.b.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_x, 0, n * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_v, 0, n * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_p, 0, n * sizeof(float)));

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
    CHECK_CUDA(cudaMalloc(&d_buffer, bufferSize));

    // r = b - Ax (initially r = b)
    CHECK_CUBLAS(cublasScopy(cublas, n, d_b, 1, d_r, 1));
    CHECK_CUBLAS(cublasScopy(cublas, n, d_r, 1, d_r_hat, 1));

    for (int i = 1; i <= max_iters; ++i) {
        CHECK_CUBLAS(cublasSdot(cublas, n, d_r_hat, 1, d_r, 1, &rho));
        
        if (i == 1) {
            CHECK_CUBLAS(cublasScopy(cublas, n, d_r, 1, d_p, 1));
        } else {
            beta = (rho / rho_prev) * (alpha / omega);
            float neg_omega = -omega;
            CHECK_CUBLAS(cublasSaxpy(cublas, n, &neg_omega, d_v, 1, d_p, 1));
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
        float neg_omega = -omega;
        CHECK_CUBLAS(cublasSaxpy(cublas, n, &neg_omega, d_t, 1, d_r, 1));

        float norm_r;
        CHECK_CUBLAS(cublasSnrm2(cublas, n, d_r, 1, &norm_r));
        if (norm_r < tolerance) break;
        
        rho_prev = rho;
    }

    // Cleanup
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecP); cusparseDestroyDnVec(vecV);
    cusparseDestroyDnVec(vecS); cusparseDestroyDnVec(vecT);
    cudaFree(d_val); cudaFree(d_row_ptr); cudaFree(d_col_ind);
    cudaFree(d_x); cudaFree(d_r); cudaFree(d_r_hat); cudaFree(d_p);
    cudaFree(d_v); cudaFree(d_s); cudaFree(d_t); cudaFree(d_b);
    cudaFree(d_buffer);
}

// Worker thread function
void gpu_worker(int device_id, std::queue<size_t>& tasks, std::mutex& mtx, const std::vector<MatrixTask>& all_tasks) {
    CHECK_CUDA(cudaSetDevice(device_id));
    
    cublasHandle_t cublas;
    cusparseHandle_t cusparse;
    CHECK_CUBLAS(cublasCreate(&cublas));
    CHECK_CUSPARSE(cusparseCreate(&cusparse));

    while (true) {
        size_t task_idx;
        {
            std::lock_guard<std::mutex> lock(mtx);
            if (tasks.empty()) break;
            task_idx = tasks.front();
            tasks.pop();
        }

        const auto& t = all_tasks[task_idx];
        if (t.type == CGS_SOLVER) {
            solve_cg(cublas, cusparse, t);
        } else {
            solve_bicgstab(cublas, cusparse, t); 
        }
    }

    cublasDestroy(cublas);
    cusparseDestroy(cusparse);
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

    std::queue<size_t> task_indices;
    for (size_t i = 0; i < tasks.size(); ++i) task_indices.push(i);
    std::mutex queue_mutex;

    std::cout << "[INFO] Processing " << tasks.size() << " tasks using " 
              << num_gpus << " GPUs (" << workers_per_gpu << " threads/GPU)...\n";

    auto start = std::chrono::steady_clock::now();

    std::vector<std::thread> workers;
    for (int i = 0; i < num_gpus; ++i) {
        for (int j = 0; j < workers_per_gpu; ++j) {
            workers.emplace_back(gpu_worker, i, std::ref(task_indices), std::ref(queue_mutex), std::cref(tasks));
        }
    }

    for (auto& w : workers) w.join();

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "\n===== NATIVE BENCHMARK COMPLETE =====\n";
    std::cout << "Tasks Processed: " << tasks.size() << "\n";
    std::cout << "Total Runtime:   " << elapsed.count() << " s\n";
    std::cout << "======================================\n";

    return 0;
}