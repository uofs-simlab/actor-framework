#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <iostream>

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

void genLaplacian1D(int *I, int *J, float *val, int N, int nz) {
    int current_nz = 0;
    I[0] = 0;
    for (int i = 0; i < N; i++) {
        if (i > 0) { J[current_nz] = i - 1; val[current_nz] = -1.5f; current_nz++; }
        J[current_nz] = i; val[current_nz] = 4.0f; current_nz++;
        if (i < N - 1) { J[current_nz] = i + 1; val[current_nz] = -0.5f; current_nz++; }
        I[i + 1] = current_nz;
    }
}

int main(int argc, char **argv) {
    int N = 10000, nz = 0;
    const float tol = 1e-4f;
    const int max_iter = 20000;

    nz = (N - 2) * 3 + 4;
    int *I = (int *)malloc(sizeof(int) * (N + 1));
    int *J = (int *)malloc(sizeof(int) * nz);
    float *val = (float *)malloc(sizeof(float) * nz);
    genLaplacian1D(I, J, val, N, nz);

    float *h_rhs = (float *)malloc(sizeof(float) * N);
    float *h_x = (float *)malloc(sizeof(float) * N);
    for (int i = 0; i < N; i++) { h_rhs[i] = 1.0f; h_x[i] = 0.0f; }

    std::cout << "\n[INFO] Starting BiCGSTAB Native GPU Benchmark (N=" << N << ")..." << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    cublasHandle_t cublasH;
    cusparseHandle_t cusparseH;
    checkCudaErrors(cublasCreate(&cublasH));
    checkCudaErrors(cusparseCreate(&cusparseH));

    int *d_row, *d_col;
    float *d_val, *d_x, *d_r, *d_r_hat, *d_p, *d_v, *d_s, *d_t;
    checkCudaErrors(cudaMalloc((void **)&d_row, (N + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_col, nz * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_val, nz * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_x, N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_r, N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_r_hat, N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_p, N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_v, N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_s, N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_t, N * sizeof(float)));

    checkCudaErrors(cudaMemcpy(d_row, I, (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_col, J, nz * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_val, val, nz * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_r, h_rhs, N * sizeof(float), cudaMemcpyHostToDevice));

    cusparseSpMatDescr_t matA;
    checkCudaErrors(cusparseCreateCsr(&matA, N, N, nz, d_row, d_col, d_val,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    float alpha = 1.0, beta = 0.0, zero = 0.0, one = 1.0, minus_one = -1.0;

    // r = b - Ax
    cusparseDnVecDescr_t vecX, vecV;
    checkCudaErrors(cusparseCreateDnVec(&vecX, N, d_x, CUDA_R_32F));
    checkCudaErrors(cusparseCreateDnVec(&vecV, N, d_v, CUDA_R_32F));

    size_t bufferSize = 0;
    void *dBuffer = NULL;
    checkCudaErrors(cusparseSpMV_bufferSize(cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &one, matA, vecX, &zero, vecV, CUDA_R_32F,
                                            CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    checkCudaErrors(cudaMalloc(&dBuffer, bufferSize));

    checkCudaErrors(cusparseSpMV(cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &one, matA, vecX, &zero, vecV, CUDA_R_32F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
    checkCudaErrors(cublasSaxpy(cublasH, N, &minus_one, d_v, 1, d_r, 1));
    checkCudaErrors(cublasScopy(cublasH, N, d_r, 1, d_r_hat, 1));

    float rho = 1.0, rho_new, alpha_bc = 1.0, omega = 1.0, beta_bc;
    float norm_sq;
    checkCudaErrors(cublasSdot(cublasH, N, d_r, 1, d_r, 1, &norm_sq));

    int k = 0;
    while (norm_sq > tol * tol && k < max_iter) {
        checkCudaErrors(cublasSdot(cublasH, N, d_r_hat, 1, d_r, 1, &rho_new));

        if (k == 0) {
            checkCudaErrors(cublasScopy(cublasH, N, d_r, 1, d_p, 1));
        } else {
            beta_bc = (rho_new / rho) * (alpha_bc / omega);
            float minus_omega = -omega;
            checkCudaErrors(cublasSaxpy(cublasH, N, &minus_omega, d_v, 1, d_p, 1));
            checkCudaErrors(cublasSscal(cublasH, N, &beta_bc, d_p, 1));
            checkCudaErrors(cublasSaxpy(cublasH, N, &one, d_r, 1, d_p, 1));
        }
        rho = rho_new;

        // v = Ap
        cusparseDnVecDescr_t vecP;
        checkCudaErrors(cusparseCreateDnVec(&vecP, N, d_p, CUDA_R_32F));
        checkCudaErrors(cusparseSpMV(cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     &one, matA, vecP, &zero, vecV, CUDA_R_32F,
                                     CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
        checkCudaErrors(cusparseDestroyDnVec(vecP));

        float dot_rv;
        checkCudaErrors(cublasSdot(cublasH, N, d_r_hat, 1, d_v, 1, &dot_rv));
        alpha_bc = rho / dot_rv;

        // s = r - alpha * v
        checkCudaErrors(cublasScopy(cublasH, N, d_r, 1, d_s, 1));
        float minus_alpha = -alpha_bc;
        checkCudaErrors(cublasSaxpy(cublasH, N, &minus_alpha, d_v, 1, d_s, 1));

        // t = As
        cusparseDnVecDescr_t vecS, vecT;
        checkCudaErrors(cusparseCreateDnVec(&vecS, N, d_s, CUDA_R_32F));
        checkCudaErrors(cusparseCreateDnVec(&vecT, N, d_t, CUDA_R_32F));
        checkCudaErrors(cusparseSpMV(cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     &one, matA, vecS, &zero, vecT, CUDA_R_32F,
                                     CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
        checkCudaErrors(cusparseDestroyDnVec(vecS));
        checkCudaErrors(cusparseDestroyDnVec(vecT));

        float dot_ts, dot_tt;
        checkCudaErrors(cublasSdot(cublasH, N, d_t, 1, d_s, 1, &dot_ts));
        checkCudaErrors(cublasSdot(cublasH, N, d_t, 1, d_t, 1, &dot_tt));
        omega = dot_ts / dot_tt;

        // x = x + alpha*p + omega*s
        checkCudaErrors(cublasSaxpy(cublasH, N, &alpha_bc, d_p, 1, d_x, 1));
        checkCudaErrors(cublasSaxpy(cublasH, N, &omega, d_s, 1, d_x, 1));

        // r = s - omega*t
        checkCudaErrors(cublasScopy(cublasH, N, d_s, 1, d_r, 1));
        float minus_omega_bc = -omega;
        checkCudaErrors(cublasSaxpy(cublasH, N, &minus_omega_bc, d_t, 1, d_r, 1));

        checkCudaErrors(cublasSdot(cublasH, N, d_r, 1, d_r, 1, &norm_sq));
        k++;
    }

    checkCudaErrors(cudaMemcpy(h_x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "[SUCCESS] BiCGSTAB Stress Test completed in " << elapsed.count() << " seconds." << std::endl;
    std::cout << "[INFO] Iterations: " << k << std::endl;
    std::cout << "[INFO] First 5 elements of solution: ";
    for(int i = 0; i < 5; ++i) std::cout << h_x[i] << " ";
    std::cout << "..." << std::endl;

    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecV);
    cudaFree(d_row); cudaFree(d_col); cudaFree(d_val);
    cudaFree(d_x); cudaFree(d_r); cudaFree(d_r_hat);
    cudaFree(d_p); cudaFree(d_v); cudaFree(d_s); cudaFree(d_t);
    cudaFree(dBuffer);
    cusparseDestroy(cusparseH);
    cublasDestroy(cublasH);
    free(I); free(J); free(val); free(h_x); free(h_rhs);

    return 0;
}