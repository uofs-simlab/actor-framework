#pragma once

#include <caf/all.hpp>
#include "caf/cuda/all.hpp"

/*
 * An actor meant to represent matrix multiply 
 * kernel for now must be supplied since we do not yet support kernel compilation
 * auto integration, it is assumed you will supply a matrix multiple kernel to this
 * actor
 * it takes in 2 mem_ptrs to the representing matrix A and matrix B,matrix size, 
 * device number, and stream id  
 * and replys with the result 
 */

namespace caf::cuda {

struct mmul_actor_state {
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
caf::behavior mmul_actor_fun(
    caf::stateful_actor<caf::cuda::mmul_actor_state>* self,
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
