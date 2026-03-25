#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(expr) do {                                  \
  cudaError_t err = (expr);                                    \
  if (err != cudaSuccess) {                                    \
    std::fprintf(stderr, "CUDA error %s:%d: %s\n",             \
                 __FILE__, __LINE__, cudaGetErrorString(err)); \
    std::exit(1);                                              \
  }                                                            \
} while (0)

extern "C" __global__
void matrixMul(const float* A, const float* B, float* C, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < N && col < N) {
    float tempVal = 0.0f;
    for (int k = 0; k < N; ++k) {
      tempVal += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = tempVal;
  }
}

int main(int /*argc*/, char** /*argv*/) {
  // Reproducible 2x2 matrix multiplication example
  const int N = 2;
  const size_t bytes = static_cast<size_t>(N) * N * sizeof(float);

  // Hard-coded inputs
  std::vector<float> hA = {
      1.0f, 2.0f,
      3.0f, 4.0f,
  };

  std::vector<float> hB = {
      5.0f, 6.0f,
      7.0f, 8.0f,
  };

  std::vector<float> hC(N * N);

  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  CUDA_CHECK(cudaMalloc(&dA, bytes));
  CUDA_CHECK(cudaMalloc(&dB, bytes));
  CUDA_CHECK(cudaMalloc(&dC, bytes));

  CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

  matrixMul<<<grid, block>>>(dA, dB, dC, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));

  bool ok = true;
  for (int row = 0; row < N && ok; ++row) {
    for (int col = 0; col < N; ++col) {
      float ref = 0.0f;
      for (int k = 0; k < N; ++k) {
        ref += hA[row * N + k] * hB[k * N + col];
      }
      float got = hC[row * N + col];
      if (std::fabs(got - ref) > 1e-3f) {
        ok = false;
        break;
      }
    }
  }

  std::cout << "Matrix A:\n";
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
        std::cout << hA[i * N + j] << " ";
    }
    std::cout << "\n";
  }

  std::cout << "Matrix B:\n";
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
        std::cout << hB[i * N + j] << " ";
    }
    std::cout << "\n";
  }

  std::cout << "Matrix C (Result):\n";
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
        std::cout << hC[i * N + j] << " ";
    }
    std::cout << "\n";
  }

  std::cout << "matmul(" << N << "x" << N << ") "
            << (ok ? "PASSED" : "FAILED") << "\n";

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));

  return ok ? 0 : 1;
}
