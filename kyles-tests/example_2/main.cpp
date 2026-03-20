// Stateful Actor Example: Matrix Multiplication with command_runner
// This example demonstrates how to use CAF's command_runner to perform matrix multiplication on the GPU.
// Answer should be 19, 22, 43, 50

#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include "vector"

// Define the command for matrix multiplication
using mmul_command = caf::cuda::command_runner<
    in<int>, // matrix A
    in<int>, // matrix B
    out<int>, // matrix C
    in<int>  // matrix size N
>;

class MatMult {
  caf::event_based_actor* self_;
  std::vector<int> A_ = {1, 2,
                         3, 4};
  std::vector<int> B_ = {5, 6,
                         7, 8};

public:
    MatMult(caf::event_based_actor* self) : self_(self) {};

    caf::behavior make_behavior() {
      return {
        [this](int N) {
          caf::cuda::manager& mgr = caf::cuda::manager::get();

          // Create the program from the CUBIN file
          auto program = mgr.create_program_from_cubin("matmul.cubin", "matrixMul");

          int THREADS = 32;
          int BLOCKS = (N + THREADS - 1) / THREADS;

          // Launch dims: 2x2 threads is enough for a 2x2 product
          caf::cuda::nd_range dim(
              BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

          // Create the arguments
          auto arg1 = caf::cuda::create_in_arg(A_);
          auto arg2 = caf::cuda::create_in_arg(B_);
          auto arg3 = caf::cuda::create_out_arg_with_size<int>(N * N);
          auto arg4 = caf::cuda::create_in_arg(N);

          // Create a command runner
          mmul_command runner;

          // Run the command
          auto result_buffer = runner.run(program, dim, self_->id(), arg1, arg2, arg3, arg4);

          // Extract the result
          std::vector<int> output = caf::cuda::extract_vector<int>(result_buffer);

          self_->println("Received result from GPU actor {}, {}, {}, {} ", 
                             output[0], output[1], output[2], output[3]);
          self_->quit();
        }
      };
    }
};


void caf_main(caf::actor_system& sys) {
    caf::scoped_actor self{sys};
    self->println("Hello, CAF!");
    caf::cuda::manager::init(sys);
    auto test_actor = self->spawn(caf::actor_from_state<MatMult>);
    self->mail(2).send(test_actor);
    self->await_all_other_actors_done();
}

CAF_MAIN()
