/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications made to track end-to-end wall time (Host allocation -> Device -> Host result).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>   // Added for high-precision host-side timing
#include <iostream>

/* Using updated (v2) interfaces to cublas */
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#ifndef checkCudaErrors
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T result, char const *const func, char const *const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d\n", file, line, static_cast<int>(result));
        exit(EXIT_FAILURE);
    }
}
#endif

/* genLaplacian1D: Generates a 1D Laplacian matrix in CSR format */
void genLaplacian1D(int *I, int *J, float *val, int N, int nz)
{
    int current_nz = 0;
    I[0] = 0;

    for (int i = 0; i < N; i++) {
        if (i > 0) {
            J[current_nz] = i - 1;
            val[current_nz] = -1.0f;
            current_nz++;
        }
        
        J[current_nz] = i;
        val[current_nz] = 2.0f;
        current_nz++;

        if (i < N - 1) {
            J[current_nz] = i + 1;
            val[current_nz] = -1.0f;
            current_nz++;
        }
        I[i + 1] = current_nz;
    }
}

int main(int argc, char **argv)
{
    int         M = 0, N = 10000, nz = 0, *I = NULL, *J = NULL;
    float      *val      = NULL;
    const float tol      = 1e-4f; 
    const int   max_iter = 20000; 
    float      *x;
    float      *rhs;
    float       a, b, na, r0, r1;
    int        *d_col, *d_row;
    float      *d_val, *d_x, dot;
    float      *d_r, *d_p, *d_Ax;
    int         k;
    float       alpha, beta, alpham1;

    // Device setup
    int devID = 0;
    checkCudaErrors(cudaSetDevice(devID));

    /* Host generation of the 1D Laplacian Operator */
    nz = (N - 2) * 3 + 4; 
    I   = (int *)malloc(sizeof(int) * (N + 1));
    J   = (int *)malloc(sizeof(int) * nz);
    val = (float *)malloc(sizeof(float) * nz);
    genLaplacian1D(I, J, val, N, nz);

    x   = (float *)malloc(sizeof(float) * N);
    rhs = (float *)malloc(sizeof(float) * N);

    for (int i = 0; i < N; i++) {
        rhs[i] = 1.0f; 
        x[i]   = 0.0f; 
    }

    std::cout << "\n[INFO] Starting GPU Benchmark: 1D Laplacian (N=" << N << ")..." << std::endl;

    // =========================================================================
    // START BENCHMARK: Captures allocations, H2D copies, Execution, and D2H copies
    // =========================================================================
    auto start_time = std::chrono::high_resolution_clock::now();

    /* 1. Get handles to the library contexts */
    cublasHandle_t cublasHandle = 0;
    checkCudaErrors(cublasCreate(&cublasHandle));

    cusparseHandle_t cusparseHandle = 0;
    checkCudaErrors(cusparseCreate(&cusparseHandle));

    /* 2. Device Memory Allocation */
    checkCudaErrors(cudaMalloc((void **)&d_col, nz * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_row, (N + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_val, nz * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_x, N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_r, N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_p, N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_Ax, N * sizeof(float)));

    /* 3. Wrap raw pointers into cuSPARSE Generic API descriptors */
    cusparseSpMatDescr_t matA = NULL;
    checkCudaErrors(cusparseCreateCsr(&matA, N, N, nz, d_row, d_col, d_val,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    cusparseDnVecDescr_t vecx = NULL;
    checkCudaErrors(cusparseCreateDnVec(&vecx, N, d_x, CUDA_R_32F));
    cusparseDnVecDescr_t vecp = NULL;
    checkCudaErrors(cusparseCreateDnVec(&vecp, N, d_p, CUDA_R_32F));
    cusparseDnVecDescr_t vecAx = NULL;
    checkCudaErrors(cusparseCreateDnVec(&vecAx, N, d_Ax, CUDA_R_32F));

    /* 4. Host-to-Device Memory Transfer (H2D) */
    checkCudaErrors(cudaMemcpy(d_col, J, nz * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_row, I, (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_val, val, nz * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_r, rhs, N * sizeof(float), cudaMemcpyHostToDevice));

    alpha   = 1.0f;
    alpham1 = -1.0f;
    beta    = 0.0f;
    r0      = 0.0f;

    /* 5. Allocate cuSPARSE internal workspace buffer */
    size_t bufferSize = 0;
    checkCudaErrors(cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha, matA, vecx, &beta, vecAx, CUDA_R_32F,
                                            CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    void *buffer = NULL;
    checkCudaErrors(cudaMalloc(&buffer, bufferSize));

    /* 6. Run Conjugate Gradient Solver Loop */
    checkCudaErrors(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecx, &beta, vecAx, CUDA_R_32F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, buffer));

    cublasSaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_r, 1);
    cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);

    k = 1;
    while (r1 > tol * tol && k <= max_iter) {
        if (k > 1) {
            b = r1 / r0;
            cublasSscal(cublasHandle, N, &b, d_p, 1);
            cublasSaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1);
        } else {
            cublasScopy(cublasHandle, N, d_r, 1, d_p, 1);
        }

        checkCudaErrors(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     &alpha, matA, vecp, &beta, vecAx, CUDA_R_32F,
                                     CUSPARSE_SPMV_ALG_DEFAULT, buffer));
        
        cublasSdot(cublasHandle, N, d_p, 1, d_Ax, 1, &dot);
        a = r1 / dot;

        cublasSaxpy(cublasHandle, N, &a, d_p, 1, d_x, 1);
        na = -a;
        cublasSaxpy(cublasHandle, N, &na, d_Ax, 1, d_r, 1);

        r0 = r1;
        cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
        k++;
    }

    /* 7. Device-to-Host Memory Transfer (D2H) */
    checkCudaErrors(cudaMemcpy(x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Force host to wait for all queued GPU operations and transfers to complete
    cudaDeviceSynchronize(); 

    auto end_time = std::chrono::high_resolution_clock::now();
    // =========================================================================
    // END BENCHMARK
    // =========================================================================

    std::chrono::duration<double> elapsed = end_time - start_time;

    // Match your actor framework's terminal presentation style
    std::cout << "[SUCCESS] Stress Test completed in " << elapsed.count() << " seconds." << std::endl;
    std::cout << "[INFO] Iterations taken to converge: " << (k - 1) << std::endl;
    std::cout << "[INFO] First 5 elements of solution: ";
    for(int i = 0; i < 5; ++i) {
        std::cout << x[i] << " ";
    }
    std::cout << "..." << std::endl;

    // Clean up GPU allocations
    if (buffer) cudaFree(buffer);
    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);
    if (matA) cusparseDestroySpMat(matA);
    if (vecx) cusparseDestroyDnVec(vecx);
    if (vecAx) cusparseDestroyDnVec(vecAx);
    if (vecp) cusparseDestroyDnVec(vecp);
    cudaFree(d_col); cudaFree(d_row); cudaFree(d_val);
    cudaFree(d_x); cudaFree(d_r); cudaFree(d_p); cudaFree(d_Ax);
    
    free(I); free(J); free(val); free(x); free(rhs);

    return 0;
}
