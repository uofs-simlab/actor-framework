#pragma once

#include <caf/all.hpp>
#include "caf/cuda/all.hpp"
#include <string>
#include <type_traits>


/*
 * An actor meant to represent matrix multiply 
 * kernel for now must be supplied since we do support kernel compilation
 * auto integration but it is via strings so you can supply own kernel to reduce needless recompilation, it is assumed you will supply a matrix multiple kernel to this
 * actor
 * it takes in 2 mem_ptrs to the representing matrix A and matrix B,matrix size, 
 * device number, and stream id  
 * and replys with the result 
 */


namespace caf::cuda {

// ------------------- Float version -------------------
inline std::string mmulNS_kernel_source_float = R"(
extern "C" __global__
void matrixMulNS(const float* A, const float* B, float* C,
                 int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float temp = 0.0f;
        for (int k = 0; k < K; ++k) {
            temp += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = temp;
    }
}
)";

// ------------------- Double version -------------------
inline std::string mmulNS_kernel_source_double = R"(
extern "C" __global__
void matrixMulNS(const double* A, const double* B, double* C,
                  int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        double temp = 0.0;
        for (int k = 0; k < K; ++k) {
            temp += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = temp;
    }
}
)";

// ------------------- Integer version -------------------
inline std::string mmulNS_kernel_source_int = R"(
extern "C" __global__
void matrixMulNS(const int* A, const int* B, int* C,
                 int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        int temp = 0;
        for (int k = 0; k < K; ++k) {
            temp += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = temp;
    }
}
)";

// ------------------- Template selector -------------------
template <typename T>
inline std::string get_mmulNS_kernel_source() {
    if constexpr (std::is_same_v<T, float>) {
        return mmulNS_kernel_source_float;
    } else if constexpr (std::is_same_v<T, double>) {
        return mmulNS_kernel_source_double;
    } else if constexpr (std::is_same_v<T, int>) {
        return mmulNS_kernel_source_int;
    } else {
        static_assert(!sizeof(T*), "Unsupported type for mmulNS kernel");
    }
}




struct mmul_actor_not_square_state {
  static inline const char* name = "mmul_actor";
  program_ptr mmul_kernel;
};

template <class T>
using mmul_async_command =
  command_runner<
    mem_ptr<T>,
    mem_ptr<T>,
    out<T>,
    in<int>
  >;

template <class T>
caf::behavior mmul_actor_NS_fun(
    caf::stateful_actor<caf::cuda::mmul_actor_not_square_state>* self,
    program_ptr mmul_kernel)
{
  using mem_t = mem_ptr<T>;
  using runner_t = mmul_async_command<T>;

  self->state().mmul_kernel = mmul_kernel;

  runner_t mmul;

  return {

    [=](mem_t matrixA,
        mem_t matrixB,
        int N,
        int device_number,
        int stream_id)
        mutable -> mem_t
    {
      program_ptr kernel = self->state().mmul_kernel;

      const int THREADS = 32;
      const int BLOCKS = (N + THREADS - 1) / THREADS;

      nd_range dims(
        BLOCKS, BLOCKS, 1,
        THREADS, THREADS, 1);

      out<T> arg3 = create_out_arg<T>(N * N);
      in<int>  arg4 = create_in_arg<int>(N);

      std::tuple<mem_t, mem_t, mem_t, mem_t> result_tuple =
        mmul.run_async(
          kernel,
          dims,
          stream_id,
          0,
          device_number,
          matrixA,
          matrixB,
          arg3,
          arg4);

      mem_t output_device_buffer = std::get<2>(result_tuple);

      return output_device_buffer;
    }

  };
}

} // namespace caf::cuda
