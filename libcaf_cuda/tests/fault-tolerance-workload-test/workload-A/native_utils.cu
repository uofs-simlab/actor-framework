#include "native_utils.hpp"
#include <cmath>
#include <chrono>

// ============================================================
// PCG Kernels
// ============================================================

__global__ void extract_diag_inv_kernel(int n, const int* row_ptr, const int* col_ind, const float* values, float* d_inv) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float diag = 1.0f;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            if (col_ind[j] == i) {
                diag = values[j];
                break;
            }
        }
        d_inv[i] = (fabsf(diag) > 1e-20f) ? 1.0f / diag : 1.0f;
    }
}

__global__ void elementwise_mul_kernel(int n, const float* a, const float* b, float* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * b[i];
}

int solve_pcg_jacobi_async(cublasHandle_t cublas, cusparseHandle_t cusparse,
                           int n, int nnz, float* d_val, int* d_row_ptr, int* d_col_ind,
                           float* d_b, float* d_x, float* d_r, float* d_p, float* d_Ap,
                           float* d_z, float* d_Dinv, cudaStream_t stream) {
    float alpha = 1.0f, beta = 0.0f, rho = 0.0f, a = 0.0f, na = 0.0f, b = 0.0f;
    float tolerance = 1e-5f;
    int max_iters = 16000;

    CHECK_CUDA(cudaMemsetAsync(d_x, 0, n * sizeof(float), stream));

    CHECK_CUBLAS(cublasSetStream(cublas, stream));
    CHECK_CUSPARSE(cusparseSetStream(cusparse, stream));

    // Extract Diagonal Inverse
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    extract_diag_inv_kernel<<<blocks, threads, 0, stream>>>(n, d_row_ptr, d_col_ind, d_val, d_Dinv);

    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecP, vecAp;
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, n, n, nnz, d_row_ptr, d_col_ind, d_val,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecP, n, d_p, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecAp, n, d_Ap, CUDA_R_32F));

    size_t bufferSize = 0;
    void* d_buffer = nullptr;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, matA, vecP, &beta, vecAp,
                                           CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMallocAsync(&d_buffer, bufferSize, stream));

    // r = b (assuming x=0), z = M^-1 * r, p = z
    CHECK_CUBLAS(cublasScopy(cublas, n, d_b, 1, d_r, 1));
    elementwise_mul_kernel<<<blocks, threads, 0, stream>>>(n, d_Dinv, d_r, d_z);
    CHECK_CUBLAS(cublasScopy(cublas, n, d_z, 1, d_p, 1));
    CHECK_CUBLAS(cublasSdot(cublas, n, d_r, 1, d_z, 1, &rho));

    int k = 0;
    float r_norm_sq = 0.0f;
    while (k < max_iters) {
        CHECK_CUSPARSE(cusparseSpMV(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, vecP, &beta, vecAp,
                                    CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer));
        float pAp, old_rho;
        CHECK_CUBLAS(cublasSdot(cublas, n, d_p, 1, d_Ap, 1, &pAp));
        a = rho / pAp;
        CHECK_CUBLAS(cublasSaxpy(cublas, n, &a, d_p, 1, d_x, 1));
        na = -a;
        CHECK_CUBLAS(cublasSaxpy(cublas, n, &na, d_Ap, 1, d_r, 1));
        
        CHECK_CUBLAS(cublasSdot(cublas, n, d_r, 1, d_r, 1, &r_norm_sq));
        if (std::sqrt(r_norm_sq) < tolerance) break;

        elementwise_mul_kernel<<<blocks, threads, 0, stream>>>(n, d_Dinv, d_r, d_z);
        old_rho = rho;
        CHECK_CUBLAS(cublasSdot(cublas, n, d_r, 1, d_z, 1, &rho));
        b = rho / old_rho;
        
        // p = z + beta * p
        CHECK_CUBLAS(cublasSscal(cublas, n, &b, d_p, 1));
        CHECK_CUBLAS(cublasSaxpy(cublas, n, &alpha, d_z, 1, d_p, 1));
        k++;
    }

    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecP));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecAp));
    CHECK_CUDA(cudaFreeAsync(d_buffer, stream));
    return k;
}

int solve_cg_async(cublasHandle_t cublas, cusparseHandle_t cusparse,
                    int n, int nnz, float* d_val, int* d_row_ptr, int* d_col_ind,
                    float* d_b, float* d_x, float* d_r, float* d_p, float* d_Ap, cudaStream_t stream) {
    float alpha = 1.0f, beta = 0.0f, r1 = 0.0f, a = 0.0f, na = 0.0f, b = 0.0f;
    float tolerance = 1e-5f;
    int max_iters = 16000;

    CHECK_CUDA(cudaMemsetAsync(d_x, 0, n * sizeof(float), stream));

    CHECK_CUBLAS(cublasSetStream(cublas, stream));
    CHECK_CUSPARSE(cusparseSetStream(cusparse, stream));

    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecP, vecAp;

    CHECK_CUSPARSE(cusparseCreateCsr(&matA, n, n, nnz, d_row_ptr, d_col_ind, d_val,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, n, d_x, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecP, n, d_p, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecAp, n, d_Ap, CUDA_R_32F));

    size_t bufferSize = 0;
    void* d_buffer = nullptr;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, matA, vecP, &beta, vecAp,
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
        float pAp, r0;
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
    CHECK_CUDA(cudaFreeAsync(d_buffer, stream));
    return k;
}

void gpu_stream_worker(int device_id, int worker_id, ThreadSafeQueue<MatrixTask>& queue,
                       std::atomic<int>& succeeded, std::atomic<int>& failed) {
    CHECK_CUDA(cudaSetDevice(device_id));
    cudaStream_t stream;
    cublasHandle_t cublas;
    cusparseHandle_t cusparse;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK_CUBLAS(cublasCreate(&cublas));
    CHECK_CUSPARSE(cusparseCreate(&cusparse));

    MatrixTask task;
    while (queue.wait_pop(task)) {
        auto pick_time = std::chrono::steady_clock::now();
        std::cout << "[WORKER " << worker_id << "] Starting: " << task.path << " (NNZ: " << task.data->nnz << ")" << std::endl;

        int n = (int)task.data->row_ptr.size() - 1;
        int nnz = task.data->nnz;
        float *d_val, *d_x, *d_r, *d_p, *d_Ap, *d_b, *d_z, *d_Dinv;
        int *d_row_ptr, *d_col_ind;

        // Allocate GPU memory once per task
        CHECK_CUDA(cudaMallocAsync(&d_val, nnz * sizeof(float), stream));
        CHECK_CUDA(cudaMallocAsync(&d_row_ptr, (n + 1) * sizeof(int), stream));
        CHECK_CUDA(cudaMallocAsync(&d_col_ind, nnz * sizeof(int), stream));
        CHECK_CUDA(cudaMallocAsync(&d_x, n * sizeof(float), stream));
        CHECK_CUDA(cudaMallocAsync(&d_r, n * sizeof(float), stream));
        CHECK_CUDA(cudaMallocAsync(&d_p, n * sizeof(float), stream));
        CHECK_CUDA(cudaMallocAsync(&d_Ap, n * sizeof(float), stream));
        CHECK_CUDA(cudaMallocAsync(&d_b, n * sizeof(float), stream));
        CHECK_CUDA(cudaMallocAsync(&d_z, n * sizeof(float), stream));
        CHECK_CUDA(cudaMallocAsync(&d_Dinv, n * sizeof(float), stream));

        // Initial Transfer
        CHECK_CUDA(cudaMemcpyAsync(d_val, task.data->values.data(), nnz * sizeof(float), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(d_row_ptr, task.data->row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(d_col_ind, task.data->col_indices.data(), nnz * sizeof(int), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(d_b, task.data->b.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream));

        int iterations = solve_cg_async(cublas, cusparse, n, nnz, d_val, d_row_ptr, d_col_ind, d_b, d_x, d_r, d_p, d_Ap, stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));

        bool success = (iterations >= 0 && iterations < MAX_ITERATIONS);

        // Fallback mechanism: If standard CG fails to converge, retry using the Jacobi Preconditioner
        if (!success) {
            std::cout << "[WORKER " << worker_id << "] CG reached max iterations (" << iterations 
                      << "). Falling back to Jacobi solver for: " << task.path << std::endl;
            iterations = solve_pcg_jacobi_async(cublas, cusparse, n, nnz, d_val, d_row_ptr, d_col_ind, d_b, d_x, d_r, d_p, d_Ap, d_z, d_Dinv, stream);
            CHECK_CUDA(cudaStreamSynchronize(stream));
            success = (iterations >= 0 && iterations < MAX_ITERATIONS);
        }

        // Clean up task memory
        CHECK_CUDA(cudaFreeAsync(d_val, stream));
        CHECK_CUDA(cudaFreeAsync(d_row_ptr, stream));
        CHECK_CUDA(cudaFreeAsync(d_col_ind, stream));
        CHECK_CUDA(cudaFreeAsync(d_x, stream));
        CHECK_CUDA(cudaFreeAsync(d_r, stream));
        CHECK_CUDA(cudaFreeAsync(d_p, stream));
        CHECK_CUDA(cudaFreeAsync(d_Ap, stream));
        CHECK_CUDA(cudaFreeAsync(d_b, stream));
        CHECK_CUDA(cudaFreeAsync(d_z, stream));
        CHECK_CUDA(cudaFreeAsync(d_Dinv, stream));

        auto finish_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(finish_time - pick_time).count();

        if (success) {
            succeeded++;
        } else {
            failed++;
        }

        record_job(task.path, task.enqueue_time, pick_time, finish_time, iterations, success);

        std::cout << "[WORKER " << worker_id << "] Done: " << task.path << " (" << iterations << " iters, " << duration << " ms)." << std::endl;
    }
    CHECK_CUBLAS(cublasDestroy(cublas));
    CHECK_CUSPARSE(cusparseDestroy(cusparse));
    CHECK_CUDA(cudaStreamDestroy(stream));
}