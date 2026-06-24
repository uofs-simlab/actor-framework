#pragma once

#include <caf/all.hpp>
#include "caf/cuda/all.hpp"

namespace caf::cuda {

// State for the vector add actor
struct vector_add_actor_state {
    static inline const char* name = "vector_add_actor";
    program_ptr vector_add_kernel;
};

// Command runner type for vector add
template <class T>
using vector_add_command = command_runner<
    mem_ptr<T>,  // input A
    mem_ptr<T>,  // input B
    out<T>,      // output C
    in<int>      // length N (always an int)
>;

// Actor behavior
template <class T>
caf::behavior vector_add_actor_fun(
    caf::stateful_actor<vector_add_actor_state>* self,
    program_ptr vector_add_kernel)
{
    using mem_t = mem_ptr<T>;
    using runner_t = vector_add_command<T>;

    self->state().vector_add_kernel = vector_add_kernel;
    runner_t runner;  // per-actor runner instance

    return {
        [=](mem_t vecA,
            mem_t vecB,
            int N,
            int device_number,
            int stream_id) mutable -> mem_t
        {
            program_ptr kernel = self->state().vector_add_kernel;

            const int THREADS = 256;
            const int BLOCKS = (N + THREADS - 1) / THREADS;

            nd_range dims(BLOCKS, 1, 1, THREADS, 1, 1);

            out<T> arg_out  = create_out_arg<T>(N);
            in<int> arg_len = create_in_arg<int>(N);  // length as int

            // Run the kernel asynchronously
            auto result_tuple = runner.run_async(
                kernel,
                dims,
                stream_id,
                0,             // shared_memory
                device_number, // device
                vecA,
                vecB,
                arg_out,
                arg_len
            );

            // Extract the output buffer (3rd element of tuple)
            mem_t output_device_buffer = std::get<2>(result_tuple);

            return output_device_buffer;
        }
    };
}

} // namespace caf::cuda
