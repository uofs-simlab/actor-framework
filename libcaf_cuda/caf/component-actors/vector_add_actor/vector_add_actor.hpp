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
    in<T>        // length N
>;

// Actor behavior
template <class T>
caf::behavior vector_add_actor_fun(
    caf::stateful_actor<vector_add_actor_state>* self,
    program_ptr vector_add_kernel)
{
    using mem_t = mem_ptr<T>;

    vector_add_command<T> runner;  // non-static, per actor
    self->state().vector_add_kernel = vector_add_kernel;

    return {
        [=](mem_t vecA,
            mem_t vecB,
            int N,
            int device_number,
            int stream_id)
        -> mem_ptr<T>
        {
            nd_range dims((N + 255) / 256, 1, 1, 256, 1, 1);

            auto arg3 = create_out_arg<T>(N);
            auto arg4 = create_in_arg<T>(N);

            auto result_tuple = runner.run_async(
                vector_add_kernel,
                dims,
                stream_id,
                0,
                device_number,
                vecA,
                vecB,
                arg3,
                arg4
            );

            // Extract mem_ptr<T> result (2nd index of result tuple)
            return extract_mem_ptr<T>(result_tuple, 2);
        }
    };
}

} // namespace caf::cuda

