#pragma once

#include <caf/all.hpp>
#include "caf/cuda/all.hpp"
#include <string>
#include <type_traits>


namespace caf::cuda {


struct mmul_actor_not_square_state {
  static inline const char* name = "mmul_actor";
  program_ptr mmul_kernel;
};

template <class T>
using mmul_async_not_square_command =
  command_runner<
    mem_ptr<T>,
    mem_ptr<T>,
    out<T>,
    in<int>,
    in<int>,
    in<int>
  >;

template <class T>
caf::behavior mmul_actor_NS_fun(
    caf::stateful_actor<caf::cuda::mmul_actor_not_square_state>* self,
    program_ptr mmul_kernel)
{
  using mem_t = mem_ptr<T>;
  using runner_t = mmul_async_not_square_command<T>;

  self->state().mmul_kernel = mmul_kernel;

  runner_t mmul;

  return {

    [=](mem_t matrixA,
        mem_t matrixB,
        int M,
	int K,
	int N,
        int device_number,
        int stream_id)
        mutable -> mem_t
    {
	    program_ptr kernel = self->state().mmul_kernel;
	    const int THREADS_X = 32;
	    const int THREADS_Y = 32;

	    const int BLOCKS_X = (N + THREADS_X - 1) / THREADS_X;
	    const int BLOCKS_Y = (M + THREADS_Y - 1) / THREADS_Y;

	    nd_range dims(
			    BLOCKS_X, BLOCKS_Y, 1,
			    THREADS_X, THREADS_Y, 1);
	    out<T> argC = create_out_arg_with_size<T>(M * N);
	    in<int>  argN = create_in_arg<int>(N);
	    in<int> argK = create_in_arg<int>(K);
	    in<int> argM = create_in_arg<int>(M);

	    auto result_tuple =
		    mmul.run_async(
				    kernel,
				    dims,
				    stream_id,
				    0,
				    device_number,
				    matrixA,
				    matrixB,
				    argC,
				    argM,
				    argK,
				    argN);

	    mem_t output_device_buffer = std::get<2>(result_tuple);

	    return output_device_buffer;
    }

  };
}

} // namespace caf::cuda
